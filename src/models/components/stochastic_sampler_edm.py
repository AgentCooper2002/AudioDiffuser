from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .utils import BrownianTreeNoiseSampler

""" Samplers Overview

Ancestral samplers: stochastic samplers, adding randomness at each sampling step. 
Including: 
    - Euler a
    - DPM2 a
    - DPM++ 2S a
    - DPM++ 2S a Karras

ODE solvers: deterministic samplers, the only randomness comes from the beginning of the sampling process.
Including:
    - Euler
    - Heun
    - LMS
    - DPM samplers
    - UniPC-p sampler

SDE solvers: stochastic samplers, adding randomness at each sampling step.
Including:


"""

def get_sigmas(sigma: float, sigma_next: float, eta=1.0) -> Tuple[float, float]:
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_up, sigma_down
    
class ADPM2Sampler(nn.Module):
    """
    Ancestral DPM sampler 2

    'DPM2 a Karras', 'sample_dpm_2_ancestral'

    Stochastic sampler
    
    """

    def __init__(self, rho: float = 1.0, num_steps: int = 50, 
                 cond_scale: float=1.0, eta: float=1.0):
        super().__init__()
        self.rho = rho
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.eta = eta

    def step(self, x: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             sigma: Tensor, 
             sigma_next: Tensor, 
             **kwargs) -> Tensor:
        
        # Sigma steps
        sigma_up, sigma_down = get_sigmas(sigma, sigma_next, self.eta)
        
        x_epis = fn(x, net=net, 
                    sigma=sigma, 
                    inference=True, 
                    cond_scale=self.cond_scale, 
                    **kwargs)
        d = (x - x_epis) / sigma

        # DPM-Solver-2
        # sigma_mid = sigma.log().lerp(sigma_down.log(), 0.5).exp()
        sigma_mid = ((sigma ** (1 / self.rho) + sigma_down ** (1 / self.rho)) / 2) ** self.rho
        x_mid = x + d * (sigma_mid - sigma)
        x_mid_epis = fn(x_mid, net=net, 
                        sigma=sigma_mid, 
                        inference=True, 
                        cond_scale=self.cond_scale, 
                        **kwargs)
        d_mid = (x_mid - x_mid_epis) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        x = x + torch.randn_like(x) * sigma_up
        return x

    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs) -> Tensor:
        
        x = sigmas[0] * noise
        
        # Denoise to sample
        for i in range(self.num_steps-1):
            x = self.step(x, fn=fn, 
                          net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          **kwargs)  # type: ignore # noqa

        return x.clamp(-1.0, 1.0)
    
class ADPMPP2SSampler(nn.Module):

    """Ancestral sampling with DPM-Solver++(2S) Karras second-order steps.

    aka "sample_dpmpp_2s_ancestral" or "DPM++ 2S a Karras"

    Stochastic sampler
    """

    def __init__(self, rho: float = 1.0, num_steps: int = 50, 
                 cond_scale: float=1.0, eta: float=1.0):
        super().__init__()
        self.rho = rho
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.eta = eta

    def step(self, x: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             sigma: float, 
             sigma_next: float, 
             **kwargs) -> Tensor:
        
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        x_epis = fn(x, net=net, 
                    sigma=sigma, 
                    inference=True, 
                    cond_scale=self.cond_scale, 
                    **kwargs)
        
        # Sigma steps
        sigma_up, sigma_down = get_sigmas(sigma, sigma_next, self.eta)

        if sigma_down == 0:
            # Euler method
            d = (x - x_epis) / sigma
            dt = sigma_down - sigma
            x = x + d * dt
        else:
        
            t, t_next = t_fn(sigma), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * x_epis
            denoised_2 = fn(x_2, net=net, 
                            sigma=sigma_fn(s), 
                            inference=True, 
                            cond_scale=self.cond_scale, 
                            **kwargs)

            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        
        if sigma_next > 0:
            x = x + torch.randn_like(x) * sigma_up
        return x

    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs) -> Tensor:
        
        x = sigmas[0] * noise
        
        # Denoise to sample
        for i in range(self.num_steps-1):
            x = self.step(x, fn=fn, 
                          net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          **kwargs)  # type: ignore # noqa

        return x.clamp(-1.0, 1.0)

class DPM2MSampler(nn.Module):

    """ 
    An implementation based on crowsonkb and hallatore from:
    https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457

    a.k.a DPM-Solver++(2M) Karras.

    Deterministic sampler
    """
    
    def __init__(self, num_steps: int = 50, cond_scale: float=1.0, reflow: bool=False):
        super().__init__()
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.reflow = reflow

    def step(self, x: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             sigma_last: float,
             sigma: float, 
             sigma_next: float, 
             old_denoised: Tensor,
             **kwargs):
        
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        denoised = fn(x, net=net, 
                      sigma=sigma, 
                      inference=True, 
                      cond_scale=self.cond_scale, 
                      **kwargs)

        if self.reflow:
            denoised = (x - denoised * sigma)
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        
        t_min = min(sigma_fn(t_next), sigma_fn(t))
        t_max = max(sigma_fn(t_next), sigma_fn(t))

        if old_denoised is None or sigma_next == 0:
            x = (t_min / t_max) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigma_last)

            h_min = min(h_last, h)
            h_max = max(h_last, h)
            r = h_max / h_min

            h_d = (h_max + h_min) / 2
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_min / t_max) * x - (-h_d).expm1() * denoised_d
       
        return x, denoised

    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, **kwargs) -> Tensor:
        
        x = sigmas[0] * noise
        
        # Denoise to sample
        old_denoised = None
        for i in range(self.num_steps):

            x, denoised = self.step(x, fn=fn, 
                                    net=net, 
                                    sigma_last=sigmas[i-1], # it's not used in the first step
                                    sigma=sigmas[i], 
                                    sigma_next=sigmas[i+1], 
                                    old_denoised = old_denoised,
                                    **kwargs)
            old_denoised = denoised

        return x.clamp(-1.0, 1.0)

class DPMPPSDESampler(nn.Module):
    '''
    
    'DPM++ SDE Karras', 'sample_dpmpp_sde'
    Stochastic sampler
    
    '''

    def __init__(self, num_steps: int = 50, cond_scale: float=1.0, eta: float=1.0, rho: float=0.5):
        super().__init__()
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.eta = eta
        self.rho = rho

    def step(self, x: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             noise_sampler: Callable,
             sigma: float, 
             sigma_next: float, 
             **kwargs) -> Tensor:
        
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        denoised = fn(x, net=net, 
                    sigma=sigma, 
                    inference=True, 
                    cond_scale=self.cond_scale, 
                    **kwargs)

        if sigma_next == 0:
            # Euler method
            d = (x - denoised) / sigma
            dt = sigma_next - sigma
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigma), t_fn(sigma_next)
            h = t_next - t
            s = t + h * self.rho
            fac = 1 / (2 * self.rho)

            # Step 1
            sd, su = get_sigmas(sigma_fn(t), sigma_fn(s), self.eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * su
            denoised_2 = fn(x_2, net=net, 
                            sigma=sigma_fn(s), 
                            inference=True, 
                            cond_scale=self.cond_scale, 
                            **kwargs)


            # Step 2
            sd, su = get_sigmas(sigma_fn(t), sigma_fn(t_next), self.eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * su

        return x
        
    def forward(self, noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs):
        
        x = sigmas[0] * noise
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

        # noise sampler for each step
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) 
        
        for i in range(self.num_steps - 1):

            x = self.step(x, fn=fn, net=net, 
                          noise_sampler=noise_sampler, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          **kwargs)

        return x.clamp(-1.0, 1.0)