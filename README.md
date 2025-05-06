# AudioDiffuser
Companion open source code for the paper "A Review on Score-based Generative Models for Audio Applications." A modular diffusion codebase designed for audio using denoising score matching formulation discussed in EDM. This repository uses hydra-lightning config management and is suitable for developping new models efficiently.

## Setup

### Install dependencies

```bash
# clone project
git clone https://github.com/AgentCooper2002/AudioDiffuser
cd AudioDiffuser

# [OPTIONAL] create conda environment
conda create -n diffaudio python=3.8
conda activate diffaudio

# install pytorch (>=2.0.1), e.g. with cuda=11.7, we have:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```

## How to run

### Run experiment and evaluation
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash ddp mixed precision
CUDA_VISIBLE_DEVICES=0,3 python src/train.py trainer=ddp.yaml trainer.devices=2 experiment=example.yaml +trainer.precision=16-mixed +trainer.accumulate_grad_batches=4
```

For RTX 4090, add `NCCL_P2P_DISABLE=1` ([verified, ref here](https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366)) otherwise, DDP will stuck.

Or train model with  single GPU resume from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=3 python src/train.py experiment=example1.yaml +trainer.precision=16-mixed ckpt_path="/path/to/ckpt/name.ckpt"
```

Or evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 python src/eval.py ckpt_path='dummy.ckpt' +trainer.precision=16 experiment=example2.yaml
```

Particularly, grid search for tuning hyperparameters during sampling:

```bash
CUDA_VISIBLE_DEVICES=2 python src/eval.py --multirun ckpt_path='ckpt.pt' +trainer.precision=16-mixed experiment=experiment.yaml model.sampler.param1=3,6,9 model.sampler.param2=1.0,1.1
```

## Code References
- [k-Diffusion by Katherine Crowson](https://github.com/crowsonkb/k-diffusion)
- [EDM by Nvdia](https://github.com/NVlabs/edm)
- [Audio Diffusion by Flavio](https://github.com/archinetai/audio-diffusion-pytorch)
- [EfficientUNet by lucidrians](https://github.com/lucidrains/imagen-pytorch)
- [UNet in ADM by openai](https://github.com/openai/guided-diffusion)
- [Unconditional diffwave by philsyn](https://github.com/philsyn/DiffWave-unconditional)


## Resources
This repo is generated with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
