import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
from julius import ResampleFrac
import random
from scipy.io import wavfile
import numpy as np
import pyloudnorm as pyln

MAX_INT16 = 32768.0

def load_audio(filepath, start=None, end=None, load_mode='torchaudio'):

    if load_mode == 'torchaudio':
        waveform, _ = torchaudio.load(filepath, frame_offset=start, 
                                      num_frames=end-start if end and start else None)
    elif load_mode == 'scipy':
        # make use of mmap to access segment from large audio files
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        _, waveform = wavfile.read(filepath, mmap=True)
        waveform = torch.from_numpy(waveform[start:end]/MAX_INT16).float().unsqueeze(0)
        
    return waveform

def load_waveform(filepath, 
                  tar_sr=None, 
                  tar_len=None,
                  start_idx=None,
                  return_start_idx=False,
                  load_mode='torchaudio'):
    """

    Args:
        filepath (str): filepath to the audio file
        tar_sr (int): target sampling rate
        tar_len (int): target length in samples
        start_idx: start index of the audio segment

    Returns:
        torch tensor: 1D waveform 
    """

    audio_metadata = torchaudio.info(filepath)
    src_len = audio_metadata.num_frames
    src_sample_rate = audio_metadata.sample_rate
    if tar_len is not None:
        # load then resample
        load_len = np.ceil(tar_len / tar_sr * src_sample_rate).astype(int)
        start = random.randint(0, max(src_len - load_len, 0)) if start_idx is None else start_idx
        waveform = load_audio(filepath, start=start, end=start + load_len, load_mode=load_mode)

    else:
        waveform = load_audio(filepath, load_mode=load_mode)
        start = 0

    # resampling if needed
    if tar_sr is not None and src_sample_rate != tar_sr:
        waveform = ResampleFrac(src_sample_rate, tar_sr)(waveform)

    # padding if needed
    src_len = waveform.shape[-1]
    if tar_len is not None and src_len < tar_len:
        waveform = F.pad(waveform, (0, tar_len-src_len), 'constant', 0)
    else:
        waveform = waveform[:, :tar_len]
    return waveform[0] if not return_start_idx else (waveform[0], start)

def add_reverb_noise(audio, reverb=None, noise=None, snr_db=0, target_len=1):
    """
    Add noise and reverberation

    Args:
        audio (_type_): _description_
        reverb (_type_): _description_
        noise (_type_): _description_
        snr_db (_type_): _description_
        target_len (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    
    noisy_speech = aF.add_noise(audio.unsqueeze(0), noise.unsqueeze(0), snr_db).squeeze(0)
    if reverb is not None:
        reverb = reverb / torch.linalg.vector_norm(reverb, ord=2) 
        reverb = reverb / reverb.abs().max()
        noisy_speech = aF.fftconvolve(noisy_speech, reverb)
    
    # noisy_speech = noisy_speech / torch.linalg.vector_norm(noisy_speech, ord=2) * torch.linalg.vector_norm(noisy_speech, ord=2)
    
    if len(noisy_speech) > target_len:
        noisy_speech = noisy_speech[:target_len]

    return noisy_speech


class HighPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, :int((nfft//2+1) * r)] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )#return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1,1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )#return_complex=False)
        x = x[:, :T].detach()
        return x
    

class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1/6, 1/3, 1/2, 2/3, 3/4, 4/5, 5/6, 1/1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        return_complex=False)
        x = x[:, :T].detach()
        return x

class SegmentMixer(nn.Module):

    """
    https://github.com/Audio-AGI/AudioSep/blob/main/data/waveform_mixers.py

    
    """
    def __init__(self, max_mix_num, lower_db, higher_db):
        super(SegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

    def __call__(self, waveforms, noise_waveforms):
        
        batch_size = waveforms.shape[0]
        noise_indices = torch.randperm(batch_size)

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(batch_size):

            segment = waveforms[n].clone()

            # random sample from noise waveforms
            noise = noise_waveforms[noise_indices[n]]
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            mix_num = random.randint(2, self.max_mix_num)
            assert mix_num >= 2

            for i in range(1, mix_num):
                next_segment = waveforms[(n + i) % batch_size]
                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            # create audio mixyure
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['segment'], data_dict['mixture']


def rescale_to_match_energy(segment1, segment2):

    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1 


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):

    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10): 
    rescaled_audio = rescale_to_match_energy(audio, reference)
    delta_loudness = random.randint(lower_db, higher_db)
    gain = np.power(10.0, delta_loudness / 20.0)

    return gain * rescaled_audio

# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = audio.squeeze(0).detach().cpu().numpy()
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = torch.from_numpy(normalized_audio).unsqueeze(0)
    
    return normalized_audio.to(device)
    