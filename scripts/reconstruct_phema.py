import sys
sys.path.append('..')

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Perform post-hoc EMA reconstruction."""

import os
import re
import copy
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import src.models.phema as phema
from typing import Any
import src
#----------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

#----------------------------------------------------------------------------
# List input pickles for post-hoc EMA reconstruction.

def list_input_pickles(
    in_dir, # Directory containing the input pickles.
    in_std=None, # Relative standard deviations of the input pickles. None = anything goes.
):
    in_std = set(in_std) if in_std is not None else None

    pkls = []
    with os.scandir(in_dir) as it:
        for e in it:
            
            m = re.fullmatch(r'ema_prof-(\d+\.\d+)_(\d+)', e.name)
            if not m or not e.is_file():
                continue
            std, nstep = float(m.group(1)), int(m.group(2))

            if in_std is not None and std not in in_std:
                continue

            pkls.append(EasyDict(path=e.path, nstep=nstep, std=std))

    pkls = sorted(pkls, key=lambda pkl: (pkl.nstep, pkl.std))
    return pkls

#----------------------------------------------------------------------------
# Perform post-hoc EMA reconstruction.
# Returns an iterable that yields dnnlib.EasyDict(out, step_idx, num_steps),
# where 'out' is a list of EasyDict(net, nstep, std, pkl_data, pkl_path)

def reconstruct_phema(
    in_pkls, # List of input pickles, expressed as dict.
    out_std, # List of relative standard deviations to reconstruct.
    out_nstep=None, # Training time of the snapshot to reconstruct. None = highest input time.
    out_prefix='recon_phema', # Output filename prefix.
    max_snapshot=8, # Maximum simultaneous reconstructions
    verbose=True, # Enable status prints?
):
    # Validate input pickles.
    if out_nstep is None:
        out_nstep = max((pkl.nstep for pkl in in_pkls), default=0)
    elif not any(out_nstep == pkl.nstep for pkl in in_pkls):
        raise click.ClickException('Reconstruction time must match one of the input pickles')
    in_pkls = [pkl for pkl in in_pkls if 0 < pkl.nstep <= pkl.nstep]
    if len(in_pkls) == 0:
        raise click.ClickException('No valid input pickles found')
    in_nstep = [pkl.nstep for pkl in in_pkls]
    in_std = [pkl.std for pkl in in_pkls]
    if verbose:
        print(f'Loading {len(in_pkls)} input pickles...')
        for pkl in in_pkls:
            print('    ' + pkl.path)

    # Determine output pickles.
    out_std = [out_std] if isinstance(out_std, float) else sorted(set(out_std))
    out_dir = os.path.dirname(in_pkls[0].path)
    num_recon_batches = (len(out_std) - 1) // max_snapshot + 1
    out_std_batches = np.array_split(out_std, num_recon_batches)
    if verbose:
        print(f'Reconstructing {len(out_std)} output pickles in {num_recon_batches} batches...')
        for i, batch in enumerate(out_std_batches):
            for std in batch:
                print(f'    batch {i}: ', end='')
                out_name = os.path.join(out_dir, out_prefix + f'-{std:.3f}_{out_nstep:07d}')
                print(out_name)

    # Return an iterable over the reconstruction steps.
    class ReconstructionIterable:
        def __len__(self):
            return num_recon_batches * len(in_pkls)

        def __iter__(self):
            # Loop over batches.
            r = EasyDict(step_idx=0, num_steps=len(self))
            for out_std_batch in out_std_batches:
                coefs = phema.solve_posthoc_coefficients(in_nstep, in_std, out_nstep, out_std_batch)
                out = [EasyDict(net=None, nstep=out_nstep, std=std) for std in out_std_batch]
                r.out = []

                # Loop over input pickles.
                for i in range(len(in_pkls)):
                    with open(in_pkls[i].path, 'rb') as f:
                        in_net = pickle.load(f).to(torch.float32)

                    # Accumulate weights for each output pickle.
                    for j in range(len(out)):
                        if out[j].net is None:
                            out[j].net = in_net
                            for pj in out[j].net.parameters():
                                pj.zero_()
                        for pi, pj in zip(in_net.parameters(), out[j].net.parameters()):
                            pj += pi * coefs[i, j]
                        for pi, pj in zip(in_net.buffers(), out[j].net.buffers()):
                            pj.copy_(pi)

                    # Finalize outputs.
                    if i == len(in_pkls) - 1:
                        for j in range(len(out)):
                            out[j].net.to(torch.float16)
                            out[j].pkl_path = out_name
                            print('Writing....', out[j].pkl_path)
                            with open(out[j].pkl_path, 'wb') as f:
                                pickle.dump(out[j].net, f)
                        r.out = out

                    # Yield results.
                    del in_net # conserve memory
                    yield r
                    r.step_idx += 1

    return ReconstructionIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of relative standard deviations.
# The special token '...' interpreted as an evenly spaced interval.
# Example: '0.01,0.02,...,0.05' returns [0.01, 0.02, 0.03, 0.04, 0.05]

def parse_std_list(s):
    if isinstance(s, list):
        return s

    # Parse raw values.
    raw = [None if v == '...' else float(v) for v in s.split(',')]

    # Fill in '...' tokens.
    out = []
    for i, v in enumerate(raw):
        if v is not None:
            out.append(v)
            continue
        if i - 2 < 0 or raw[i - 2] is None or raw[i - 1] is None:
            raise click.ClickException("'...' must be preceded by at least two floats")
        if i + 1 >= len(raw) or raw[i + 1] is None:
            raise click.ClickException("'...' must be followed by at least one float")
        if raw[i - 2] == raw[i - 1]:
            raise click.ClickException("The floats preceding '...' must not be equal")
        approx_num = (raw[i + 1] - raw[i - 1]) / (raw[i - 1] - raw[i - 2]) - 1
        num = round(approx_num)
        if num <= 0:
            raise click.ClickException("'...' must correspond to a non-empty interval")
        if abs(num - approx_num) > 1e-4:
            raise click.ClickException("'...' must correspond to an evenly spaced interval")
        for j in range(num):
            out.append(raw[i - 1] + (raw[i - 1] - raw[i - 2]) * (j + 1))

    # Validate.
    out = sorted(set(out))
    if not all(0.000 < v < 0.289 for v in out):
        raise click.ClickException('Relative standard deviation must be positive and less than 0.289')
    return out

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--indir', 'in_dir', help='Directory containing the input pickles', metavar='DIR', type=str, required=True)
@click.option('--instd', 'in_std', help='Filter inputs based on standard deviations', metavar='LIST', type=parse_std_list, default=None)

@click.option('--outstd', 'out_std', help='List of desired relative standard deviations', metavar='LIST', type=parse_std_list, required=True)
@click.option('--outnstep', 'out_nstep', help='Training time of the snapshot to reconstruct', metavar='NSTEP', type=click.IntRange(min=1), default=None)

@click.option('--batch', 'training_batch_size', help='batch size used during training', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--snapshot', 'max_snapshot', help='Maximum simultaneous reconstructions', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)

def main(in_dir, in_std, training_batch_size, out_nstep, **kwargs):
    """Perform post-hoc EMA reconstruction.

    Examples:

    \b
    # Reconstruct a new EMA profile with std=0.150
    python reconstruct_phema.py --indir=raw-snapshots/edm2-img512-xs \\
        --outdir=out --outstd=0.150

    \b
    # Reconstruct a set of 31 EMA profiles, streaming over the input data 4 times
    python reconstruct_phema.py --indir=raw-snapshots/edm2-img512-xs \\
        --outdir=out --outstd=0.010,0.015,...,0.250 --batch=8

    \b
    # Perform reconstruction for the latest snapshot of a given training run
    python reconstruct_phema.py --indir=training-runs/00000-edm2-img512-xs \\
        --outdir=out --outstd=0.150
    """

    in_pkls = list_input_pickles(in_dir=in_dir, in_std=in_std)
    rec_iter = reconstruct_phema(in_pkls=in_pkls, out_nstep=out_nstep, **kwargs)
    for _r in tqdm.tqdm(rec_iter, unit='step'):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------