from typing import Any, Optional
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from .utils import spec_fwd, spec_back
from .phema import PowerFunctionEMA, TraditionalEMA
import copy, pickle

class DiffUnetComplexModule(LightningModule):

    def __init__(
        self,
        spec_abs_exponent: float,
        spec_factor: float,
        net: torch.nn.Module,
        noise_scheduler: torch.nn.Module,
        noise_distribution: torch.nn.Module,
        sampler: torch.nn.Module,
        diffusion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        generated_frame_length: int,
        generated_frequency: int,
        generated_sample_class: int,
        audio_sample_rate: int,
        hop_length: int,
        n_fft: int,
        norm_wav: bool = False,
        center: bool = True,
        use_ema: bool = True,
        use_phema: bool = False,
        num_ema_snapshot_item: Optional[int] = 96000,
        total_test_samples: Optional[int] = None,
        ema_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.spec_abs_exponent = spec_abs_exponent
        self.spec_factor = spec_factor
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, center=True)
        self.window = torch.hann_window(self.stft_args['n_fft'], periodic=True).to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # diffusion components
        self.net = net
        self.use_ema = use_ema
        self.use_phema = use_phema
        self.cur_nitem = 0
        self.num_ema_snapshot_item = num_ema_snapshot_item
        self.ema_ckpt_path = ema_ckpt_path

        self.sampler = sampler
        self.diffusion = diffusion
        self.noise_distribution = noise_distribution # for training
        self.noise_scheduler = noise_scheduler()     # for sampling
        self.generated_frame_length = generated_frame_length
        self.generated_frequency = generated_frequency
        self.generated_sample_class = generated_sample_class
        
        # generation
        self.total_test_samples = total_test_samples
        self.audio_sample_rate = audio_sample_rate
        self.norm_wav = norm_wav
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    @torch.no_grad()
    def synthesize_from_noise(self, initial_noise, target_class, ema_model=None):

        # synthesize "complex" spec
        pcomplex_spec = self.sampler(initial_noise, classes=target_class, 
                                     fn=self.diffusion.denoise_fn, 
                                     net=self.net, 
                                     sigmas=self.noise_scheduler.to(self.device))
        pcomplex_spec = torch.permute(pcomplex_spec, (0, 2, 3, 1)).contiguous()
        complex_spec = torch.view_as_complex(pcomplex_spec)[:, None, :, :]
        complex_spec = spec_back(complex_spec,
                                 spec_abs_exponent=self.spec_abs_exponent, 
                                 spec_factor=self.spec_factor)
        
        # convert to waveform
        audio_sample = torch.istft(complex_spec.squeeze(1), 
                                   window=self.window.to(self.device), 
                                   normalized=True, **self.stft_args)
        audio_sample = audio_sample.cpu()
        
        return audio_sample

    def forward(self, x: torch.Tensor):
        # predict noise
        
        audio_classes = x['label'] # kwargs
        audio = x['audio'].to(next(self.net.parameters()).dtype)
        audio_spec = torch.stft(audio, window=self.window.to(self.device), 
                                normalized=True, return_complex=True, **self.stft_args)

        # Convert real and imaginary parts of x into two channel dimensions
        audio_spec = spec_fwd(audio_spec, spec_abs_exponent=self.spec_abs_exponent, 
                              spec_factor=self.spec_factor).unsqueeze(1)
        audio_spec = torch.cat((audio_spec.real, audio_spec.imag), dim=1)

        # Sample amount of noise to add for each batch element
        sigmas = self.noise_distribution(num_samples=audio_spec.shape[0], 
                                         device=audio_spec.device)
        
        # compute loss
        loss = self.diffusion(audio_spec, self.net, 
                              classes=audio_classes, 
                              sigmas=sigmas)
        return loss.mean()
    
    def on_fit_start(self):

        if self.use_ema and self.use_phema:
            self.ema_prof = PowerFunctionEMA(self.net.to(self.device), stds=[0.050, 0.100])
        elif self.use_ema:
            self.ema_prof = TraditionalEMA(self.net.to(self.device), halflife_Mimg=0.3, rampup_ratio=0.09)


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        loss = self.forward(batch)
        return loss

    def training_step(self, batch: Any, batch_idx: int):

        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("seen items", self.cur_nitem * 1.0, on_step=True, prog_bar=True, sync_dist=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        
        if self.use_ema:
            batch_size = batch[list(batch.keys())[0]].shape[0]
            if int(self.cur_nitem) % self.num_ema_snapshot_item == 0 and self.trainer.global_rank == 0 and self.global_step > 0:
                ema_list = self.ema_prof.get()
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    ema_snapshot_path = os.path.join(self.logger.save_dir, 'ema_snapshots')
                    os.makedirs(ema_snapshot_path, exist_ok=True)
                    ema_snapshot = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                    with open(os.path.join(ema_snapshot_path, f'ema_prof{ema_suffix}_{self.global_step}'), 'wb') as f:
                        pickle.dump(ema_snapshot, f)
                    del ema_snapshot

            # update EMA and training state
            self.cur_nitem += batch_size
            self.ema_prof.update(self.cur_nitem, batch_size)
        return {"loss": loss}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass
    
    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int):
        
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}
    
    @torch.no_grad()
    def on_validation_epoch_end(self):

        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True, sync_dist=True)
        target_classes = list(range(self.generated_sample_class)) if self.generated_sample_class > 1 else [0]

        with torch.no_grad():
            target_class = torch.from_numpy(np.random.choice(target_classes, 1).astype(int)).to(self.device)
            
            # input data
            initial_noise = torch.randn((1, 2, self.n_fft//2+1, self.generated_frame_length), device=self.device)

            audio_sample = self.synthesize_from_noise(initial_noise, target_class)
        
        if self.trainer.is_global_zero:
            audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
            os.makedirs(audio_save_dir, exist_ok=True)
            audio_path = os.path.join(audio_save_dir, 'val_' + str(target_class[0].item()) + '_' + str(self.global_step) + '.wav')
            torchaudio.save(audio_path, audio_sample, self.audio_sample_rate)

    def test_step(self, batch: Any, batch_idx: int):
        # loss = self.model_step(batch)

        # # update and log metrics
        # self.test_loss(loss)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return {"loss": loss}
        pass

    def on_test_epoch_end(self):

        test_batch = self.trainer.datamodule.batch_size # 8 #self.trainer.batch_size
        audio_dur = 1

        iteration = self.total_test_samples // test_batch
        test_sample_folder = os.path.join(self.logger.save_dir, 'test_samples')

        # override neural network weights with EMA
        if self.ema_ckpt_path is not None:
            print('Loading EMA weights....................')
            with open(self.ema_ckpt_path, 'rb') as f:
                self.net = pickle.load(f).to(self.dtype).to(self.device)

        print('Generating test samples....................')
        
        os.makedirs(test_sample_folder, exist_ok=True)
        
        with torch.no_grad():
            
            for i in tqdm(range(iteration)):

                if self.generated_sample_class > 1:
                    target_class = torch.from_numpy((np.arange(test_batch) % self.generated_sample_class).astype(int)).to(self.device)
                else:
                    target_class = torch.from_numpy(0*np.ones(test_batch).astype(int)).to(self.device)
                
                # input data
                initial_noise = torch.randn((test_batch, 2, self.n_fft//2+1, self.generated_frame_length), device=self.device)

                audio_samples = self.synthesize_from_noise(initial_noise, target_class)

                for j in range(audio_samples.shape[0]):
                    audio_filename = 'test_'+str(target_class[j].item())+'_'+str(i*test_batch+j)+'.wav'
                    audio_path = os.path.join(test_sample_folder, audio_filename)
                    torchaudio.save(audio_path, audio_samples[j, :int(audio_dur*self.audio_sample_rate)].unsqueeze(0), 
                                    self.audio_sample_rate, bits_per_sample=16)
                
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

