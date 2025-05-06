""" Diffusion Classes """

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from torch import Tensor
from .utils import extend_dim, clip, to_batch
from abc import abstractmethod

EPSI = 1e-7

class Diffusion(nn.Module):

    def __init__(
        self,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
    
    @abstractmethod
    def loss_weight(self):
        pass
    
    @abstractmethod
    def get_scale_weights(self):
        pass
    
    def denoise_fn(
        self,
        x_noisy: Tensor,
        net: nn.Module = None,
        inference: bool = False,
        cond_scale: float = 1.0,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs) -> Tensor:

        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        
        # Predict network output
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy.ndim)

        # cfg interpolation during inference, skip during training
        if inference:
            x_pred = net(c_in*x_noisy, c_noise, cond_drop_prob=0., **kwargs)
            
            if cond_scale != 1.0:
                null_logits = net(c_in*x_noisy, c_noise, cond_drop_prob=1., **kwargs)
                x_pred = null_logits + (x_pred - null_logits) * cond_scale
        
        else:
            x_pred = net(c_in*x_noisy, c_noise, **kwargs)

        # eq.7
        x_denoised = c_skip * x_noisy + c_out * x_pred
        
        # Clips in [-1,1] range, with dynamic thresholding if provided
        return clip(x_denoised, dynamic_threshold=self.dynamic_threshold)
    
    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        device = x.device
        sigmas_padded = extend_dim(sigmas, dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)

        # Compute denoised values
        x_noisy = x + sigmas_padded * noise
        if 'x_mask' in kwargs:
            loss_mask = torch.ones(x.size(), device=device) * kwargs['x_mask'] + torch.ones(x.size(), device=device) * (~kwargs['x_mask'])*0.01
        else: 
            loss_mask = torch.ones(x.size(), device=device)

        x_denoised = self.denoise_fn(x_noisy=x_noisy, 
                                     net=net, sigmas=sigmas, 
                                     inference=inference, 
                                     cond_scale=cond_scale,
                                     **kwargs)
        
        # noise level weighted loss (weighted eq.2)
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses * loss_mask, "b ... -> b", "sum")
        x_dim = list(range(len(x.shape)))
        losses = losses * self.loss_weight(sigmas) / torch.sum(torch.ones(x.size(), device=device), dim=tuple(x_dim[1:]))

        return losses # loss shape [B,]
    
class VEDiffusion(Diffusion):
    def __init__(
        self,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
        
    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1
        c_noise = (0.5 * sigmas).log()
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = 1
        c_out = sigmas
        c_in = 1
        
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return 1 / (sigmas**2)
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
                   
        return super().denoise_fn(x_noisy, net, 
                                  inference, cond_scale, 
                                  sigmas, sigma,
                                  **kwargs)


class VPDiffusion(Diffusion):
    """VP Diffusion Models formulated by EDM"""

    def __init__(
        self,
        beta_min: float, 
        beta_d: float,
        M: float,
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.M = M
        self.dynamic_threshold = dynamic_threshold

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return 1 / sigmas ** 2
    
    def t_to_sigma(self, t):
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def sigma_to_t(self, sigmas):
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigmas ** 2).log()).sqrt() - self.beta_min) / self.beta_d
    
    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1 
        c_noise = (self.M - 1) * self.sigma_to_t(sigmas)
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = 1
        c_out = - sigmas
        c_in = 1 / (sigmas ** 2 + 1).sqrt()
        return c_skip, c_out, c_in, c_noise
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
                   
        return super().denoise_fn(x_noisy, net, 
                                  inference, cond_scale, 
                                  sigmas, sigma,
                                  **kwargs)

    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        # # Sample amount of noise to add for each batch element
        sigmas = self.t_to_sigma(sigmas)
        sigmas_padded = extend_dim(sigmas, dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)

        # Compute denoised values
        x_noisy = x + sigmas_padded * noise
        x_denoised = self.denoise_fn(x_noisy, net, 
                                     sigmas=sigmas,
                                     inference=inference, 
                                     cond_scale=cond_scale,
                                     **kwargs)
        

        if 'x_mask' in kwargs:
            loss_mask = torch.ones(x.size(), device=x.device) * kwargs['x_mask'] + torch.ones(x.size(), device=x.device) * (~kwargs['x_mask'])*0.1
        else: 
            loss_mask = torch.ones(x.size(), device=x.device)

        # noise level weighted loss (weighted eq.2)
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses * loss_mask, "b ... -> b", "sum")
        losses = losses * self.loss_weight(sigmas) / torch.sum(torch.ones(x.size(), device=x.device), dim=(1,2,3))

        return losses

class EluDiffusion(Diffusion):
    """Elucidated Diffusion Models(EDM): https://arxiv.org/abs/2206.00364"""

    def __init__(
        self,
        sigma_data: float,  # data distribution
        dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor, ex_dim: int) -> Tuple[Tensor, ...]:

        # preconditioning equations in table.1 
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = extend_dim(sigmas, dim=ex_dim)
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
                   
        return super().denoise_fn(x_noisy, net, 
                                  inference, cond_scale, 
                                  sigmas, sigma,
                                  **kwargs)
    
class VDiffusion(Diffusion):
    
    """ 
    v-diffusion predicts a combination of the noise and data
    
    """

    def __init__(
        self,
        dynamic_threshold: float = 0.0,
        logsnr_min = -15, 
        logsnr_max = 15,
        shift = 0.0,
        for_edm: bool = False
    ):
        super().__init__()
        self.dynamic_threshold = dynamic_threshold
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.shift = shift
        self.for_edm = for_edm

    def shifted_cosine_transform(self, t: Tensor) -> Tensor:
        t_min = math.atan(math.exp(-0.5 * self.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.logsnr_min))
        return -2 * (torch.tan(t_min + t * (t_max - t_min)).log()) + 2 * self.shift
    
    def sigma_to_logsnr(self, sigma):
        return -2 * sigma.log()
    
    def v_to_x0(self, x_noisy: Tensor, v_pred: Tensor, alphat: Tensor, sigmat: Tensor) -> Tensor:
        return alphat * x_noisy - sigmat * v_pred
    
    def v_to_eps(self, x_noisy: Tensor, v_pred: Tensor, alphat: Tensor, sigmat: Tensor) -> Tensor:
        return sigmat * x_noisy + alphat * v_pred

    def denoise_fn(
        self,
        x_noisy: Tensor,
        net: nn.Module = None,
        inference: bool = False,
        cond_scale: float = 1.0,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs) -> Tensor:

        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        if self.for_edm:
            sigmat = torch.sqrt(torch.sigmoid(-self.sigma_to_logsnr(sigmas)))
            alphat = torch.sqrt(torch.sigmoid(self.sigma_to_logsnr(sigmas)))
            x_noisy = x_noisy * alphat
            sigmas = self.sigma_to_logsnr(sigmas)

        # cfg interpolation during inference, skip during training
        if inference:
            v_pred = net(x_noisy, sigmas, cond_drop_prob=0., **kwargs)
            
            if cond_scale != 1.0:
                null_logits = net(x_noisy, sigmas, cond_drop_prob=1., **kwargs)
                v_pred = null_logits + (v_pred - null_logits) * cond_scale

        else:
            v_pred = net(x_noisy, sigmas, **kwargs)

        return self.v_to_x0(x_noisy, v_pred, alphat, sigmat) if self.for_edm else v_pred
    
    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs) -> Tensor:

        device = x.device

        logsnr_t = self.shifted_cosine_transform(sigmas)
        alpha_t = extend_dim(torch.sqrt(torch.sigmoid(logsnr_t)), dim=x.ndim)
        sigma_t = extend_dim(torch.sqrt(torch.sigmoid(-logsnr_t)), dim=x.ndim)

        # Add noise to input
        noise = torch.randn_like(x)

        x_noisy = alpha_t * x + sigma_t * noise
        
        # Compute denoised values
        if 'x_mask' in kwargs:
            loss_mask = torch.ones(x.size(), device=device) * kwargs['x_mask'] + torch.ones(x.size(), device=device) * (~kwargs['x_mask'])*0.1
        else: 
            loss_mask = torch.ones(x.size(), device=device)

        v_pred = self.denoise_fn(x_noisy, net, sigmas=logsnr_t, inference=inference, cond_scale=cond_scale, **kwargs)

        # Compute v-objective target
        eps_pred = self.v_to_eps(x_noisy, v_pred, alpha_t, sigma_t)

        snr = torch.exp(logsnr_t).clamp_(max = 5)
        weight = extend_dim(1 / (1 + snr), dim=x.ndim)

        # Compute loss (need mask fixing)
        losses = F.mse_loss(eps_pred, noise, reduction="none") 
        losses = reduce(weight * losses * loss_mask, "b ... -> b", "sum")
        x_dim = list(range(len(x.shape)))
        losses = losses / torch.sum(torch.ones(x.size(), device=device), dim=tuple(x_dim[1:]))
        return losses
    
class ReFlow(nn.Module):
    # Rectified flow training
    # Reference:
    #   https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py

    def __init__(
        self,
        for_edm: bool = False,
    ):
        super().__init__()
        self.for_edm = for_edm
    
    def sigma_to_t(self, t):
        return t / (t + 1)

    def v_to_x0(self, x_noisy: Tensor, v_pred: Tensor, sigmas: Tensor) -> Tensor:
        return x_noisy - v_pred * sigmas
    
    def v_to_eps(self, x_noisy: Tensor, v_pred: Tensor, sigmas: Tensor) -> Tensor:
        return x_noisy + v_pred * (1 - sigmas)
    
    def denoise_fn(self, x_noisy: Tensor,
                   net: nn.Module = None,
                   inference: bool = False,
                   cond_scale: float = 1.0,
                   sigmas: Optional[Tensor] = None,
                   sigma: Optional[float] = None,
                   **kwargs) -> Tensor:
        # denoise means an EDM wrapper for ReFlow sampling when for_edm is True

        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        if self.for_edm:
            sigmas = self.sigma_to_t(sigmas)
            x_noisy = x_noisy * (1 - sigmas)

        # cfg interpolation during inference, skip during training
        if inference:
            x_pred = net(x_noisy, sigmas, cond_drop_prob=0., **kwargs)

            if cond_scale != 1.0:
                null_logits = net(x_noisy, sigmas, cond_drop_prob=1., **kwargs)
                x_pred = null_logits + (x_pred - null_logits) * cond_scale
        
        else:
            x_pred = net(x_noisy, sigmas, **kwargs)

        if self.for_edm:  # output x0 prediction
            x_pred = self.v_to_x0(x_noisy, x_pred, sigmas)
        
        return x_pred

    def forward(self, x: Tensor, 
                net: nn.Module, 
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs):

        # EDM wrapper for ReFlow training
        t = sigmas
        t_padded = extend_dim(sigmas, dim=x.ndim)
        z1 = torch.randn_like(x)
        zt = (1 - t_padded) * x + t_padded * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)

        vtheta = self.denoise_fn(zt, net, sigmas=t,
                                 inference=inference, 
                                 cond_scale=cond_scale,
                                 **kwargs)
        
        losses = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        return losses
