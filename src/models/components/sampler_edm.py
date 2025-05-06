from math import sqrt
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from scipy import integrate

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

class VESampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) VE sampler, stochastic sampler

    """
    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float('inf'),
        s_churn: float = 200,
        s_noise: float = 1,
        num_steps: int = 200,
        cond_scale: float = 1.0,
        use_heun: bool = True
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_churn = s_churn
        self.s_noise = s_noise
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.use_heun = use_heun
    
    def step(self, x: Tensor, 
             fn: Callable, net: nn.Module, 
             t: float, t_next: float, 
             gamma: float, 
             **kwargs) -> Tensor:

        # Increase noise temporarily.
        t_hat = self.sigma_to_t((self.t_to_sigma(t) + gamma * self.t_to_sigma(t)))
        x_hat = x + (self.t_to_sigma(t_hat)**2 - self.t_to_sigma(t)**2).clip(min=0).sqrt() * self.s_noise * torch.randn_like(x)

        # Euler step.
        denoised_cur = fn(x_hat, net=net, 
                          sigma=self.t_to_sigma(t_hat), 
                          inference=True, 
                          cond_scale=self.cond_scale, 
                          **kwargs)
        
        d = (self.t_to_sigma_deriv(t_hat)/self.t_to_sigma(t_hat))*x_hat - self.t_to_sigma_deriv(t_hat)/self.t_to_sigma(t_hat) * denoised_cur
        h = t_next - t_hat
        x_next = x_hat + h * d
        
        # Apply 2nd order correction.
        if t_next != 0 and self.use_heun:
            t_prime = t_hat + h
            
            denoised_prime = fn(x_next, net=net, 
                                sigma=self.t_to_sigma(t_prime), 
                                inference=True, 
                                cond_scale=self.cond_scale, 
                                **kwargs)
            d_prime = self.t_to_sigma_deriv(t_prime) / self.t_to_sigma(t_prime) * x_next - self.t_to_sigma_deriv(t_prime) / self.t_to_sigma(t_prime) * denoised_prime
            x_next = x_hat + 1/2 * h * (d + d_prime)
        
        return x_next

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return t.sqrt()
    
    def t_to_sigma_deriv(self, t: Tensor) -> Tensor:
        return 0.5 / t.sqrt()

    def sigma_to_t(self, sigma: Tensor) -> Tensor:
        return sigma ** 2
    
    def forward(self, noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor,
                **kwargs) -> Tensor:
        
        ts = self.sigma_to_t(sigmas)
        ts = torch.cat([ts, torch.zeros_like(ts[:1])]) # t_N = 0
        
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )

        # Denoise to sample
        x = noise * sigmas[0]
        for i in range(self.num_steps):
            x = self.step(x, fn=fn, 
                          net=net, 
                          gamma=gammas[i],
                          t=ts[i], 
                          t_next=ts[i+1],
                          **kwargs)
        return x.clamp(-1.0, 1.0)
    
class VPSampler(nn.Module):
    """ 
    EDM version (https://arxiv.org/abs/2206.00364) VP sampler in Algorithm 1:
    """
    def __init__(
        self,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        s_churn: float = 200.0,
        s_noise: float = 1.0,
        s_min: float = 0.0,
        s_max: float = float('inf'),
        num_steps: int = 200,
        cond_scale: float = 1.0,
        use_heun: bool = True
    ):
        super().__init__()
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.s_noise = s_noise
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.use_heun = use_heun

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1) ** 0.5

    def sigma_to_t(self, sigma: Tensor) -> Tensor:
        return ((self.beta_min ** 2 + 2 * self.beta_d * (sigma ** 2 + 1).log()).sqrt() - self.beta_min) / self.beta_d

    def t_to_sigma_deriv(self, t: Tensor) -> Tensor:
        # sigma'(t) = 0.5 / sqrt(t)
        return 0.5 * (self.beta_min + self.beta_d * t) * (self.t_to_sigma(t) + 1 / self.t_to_sigma(t))

    def scale(self, t: Tensor) -> Tensor:
        return 1 / (1 + self.t_to_sigma(t) ** 2).sqrt()
    
    def scale_deriv(self, t: Tensor) -> Tensor:
        return -self.t_to_sigma(t) * self.t_to_sigma_deriv(t) * (self.scale(t) ** 3)
        
    def step(self, x: Tensor, 
             fn: Callable, net: nn.Module, 
             t: float, t_next: float, 
             gamma: float, 
             **kwargs) -> Tensor:

        # Increase noise temporarily.
        t_hat = self.sigma_to_t((self.t_to_sigma(t) + gamma * self.t_to_sigma(t)))
        x_hat = self.scale(t_hat) / self.scale(t) * x + (self.t_to_sigma(t_hat)**2-self.t_to_sigma(t)**2).clip(min=0).sqrt() * self.scale(t_hat) * self.s_noise * torch.randn_like(x)

        # Euler step.
        denoised_cur = fn(x_hat / self.scale(t_hat), 
                          net=net, sigma=self.t_to_sigma(t_hat), 
                          inference=True, 
                          cond_scale=self.cond_scale, **kwargs)
        
        d = (self.t_to_sigma_deriv(t_hat)/self.t_to_sigma(t_hat) + self.scale_deriv(t_hat)/self.scale(t_hat))*x_hat - self.t_to_sigma_deriv(t_hat) * self.scale(t_hat) / self.t_to_sigma(t_hat) * denoised_cur
        h = t_next - t_hat
        x_next = x_hat + h * d
        
        # Apply 2nd order correction.
        if t_next != 0 and self.use_heun:
            t_prime = t_hat + h
            
            denoised_prime = fn(x_next / self.scale(t_prime), 
                                net=net, sigma=self.t_to_sigma(t_prime), 
                                inference=True, cond_scale=self.cond_scale, 
                                **kwargs)
            d_prime = (self.t_to_sigma_deriv(t_prime) / self.t_to_sigma(t_prime) + self.scale_deriv(t_prime) / self.scale(t_prime)) * x_next - self.t_to_sigma_deriv(t_prime) * self.scale(t_prime) / self.t_to_sigma(t_prime) * denoised_prime
            x_next = x_hat + 1/2 * h * (d + d_prime)
        
        return x_next
        
    def forward(self, noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, # actually t_raw
                **kwargs) -> Tensor:
        
        ts = self.sigma_to_t(sigmas)
        ts = torch.cat([ts, torch.zeros_like(ts[:1])]) # t_N = 0

        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_min) & (sigmas <= self.s_max),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )
        
        # Denoise to sample
        x = noise * sigmas[0] * self.scale(ts[0])
        for i in range(self.num_steps):
            
            x = self.step(x, fn=fn, net=net, 
                          t=ts[i], 
                          t_next=ts[i+1], 
                          gamma=gammas[i], 
                          **kwargs)

        return x

class EDMAlphaSampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) deterministic sampler Algo 3
    with general Runge-Kutta method, with alpha=1, it is exactly Heun method

    Deterministic sampler
    
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_steps: int = 50,
        cond_scale: float = 1.0,
        use_heun: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.use_heun = use_heun

    @torch.no_grad()
    def step(self, x: Tensor, 
             fn: Callable, net: nn.Module, 
             sigma: float, sigma_next: float, 
             **kwargs) -> Tensor:
        
        """One step of EDM alpha sampler"""
        # Select temporarily increased noise level
        h = sigma_next - sigma

        # Evaluate denoised value at sigma
        denoised_cur =  fn(x, net=net, 
                           sigma=sigma, 
                           inference=True, 
                           cond_scale=self.cond_scale, 
                           **kwargs)
        d = (x - denoised_cur) / sigma

        sigma_p = sigma + self.alpha * h
        if sigma_p != 0 and self.use_heun:
            # Second order correction
            x_p = x + self.alpha * h * d
            denoised_p = fn(x_p, net=net, sigma=sigma_p, 
                            inference=True, 
                            cond_scale=self.cond_scale, 
                            **kwargs)
            d_p = (x_p - denoised_p) / sigma_p
            x_next = x + h * ((1 - 0.5/self.alpha) * d + 0.5/self.alpha * d_p)
        else:
            x_next = x + h * d
            
        return x_next
    
    def forward(self, noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs) -> Tensor:
        
        # pay attention to this step
        x = sigmas[0] * noise

        # Denoise to sample
        for i in range(self.num_steps-1):
            x = self.step(x, fn=fn, net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          **kwargs)
            
        return x

class EDMSampler(nn.Module):
    """ 
    EDM (https://arxiv.org/abs/2206.00364) stochastic sampler:
    s_churn=40 s_noise=1.003, s_tmin=0.05, s_tmax=50 
    when setting s_churn = 0, it amounts to DDIM

    with Heun's sampler, the NFE is around the twice the number of steps. 

    Hybrid sampler
    
    """

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float('inf'),
        s_churn: float = 150.0,
        s_noise: float = 1.04,
        num_steps: int = 200,
        cond_scale: float = 1.0,
        use_heun: bool = True
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.use_heun = use_heun

    def step(self, x: Tensor, 
             fn: Callable,
             net: nn.Module, 
             sigma: float, 
             sigma_next: float, 
             gamma: float, 
             **kwargs) -> Tensor:
        
        """One step of EDM sampler"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma if gamma > 0 else sigma

        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** 0.5 * epsilon if gamma > 0 else x

        # Evaluate ∂x/∂sigma at sigma_hat
        denoised_cur =  fn(x_hat, net=net, 
                           sigma=sigma_hat, inference=True, 
                           cond_scale=self.cond_scale, 
                           **kwargs)
        d = (x_hat - denoised_cur) / sigma_hat

        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d

        # Second order correction
        if sigma_next != 0 and self.use_heun:
            denoised_next = fn(x_next, net=net, 
                               sigma=sigma_next, 
                               inference=True, 
                               cond_scale=self.cond_scale, 
                               **kwargs)
            d_prime = (x_next - denoised_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_prime)

        return x_next
    
    def forward(self, noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor,
                **kwargs) -> Tensor:
        
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])]) # t_N = 0

        # pay attention to this step
        x = sigmas[0] * noise

        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )

        # Denoise to sample
        for i in range(self.num_steps):
            x = self.step(x, fn=fn, net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          gamma=gammas[i],
                          **kwargs)
            
        return x



class DPM2Sampler(nn.Module):
    """
    DPM sampler 2 using EDM

    a.k.a 'DPM2 Karras', 'sample_dpm_2'

    Hybrid sampler (S_churn is used)
    
    """

    def __init__(self, rho: float = 2.0, 
                 num_steps: int = 50, 
                 cond_scale: float=1.0, 
                 s_tmin: float = 0,
                 s_tmax: float = float('inf'),
                 s_churn: float = 150.0,
                 s_noise: float = 1.04):
        
        super().__init__()
        self.rho = rho
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(self, x: Tensor, 
             fn: Callable, net: nn.Module, 
             sigma: float, sigma_next: float, 
             gamma: float, 
             **kwargs) -> Tensor:
        
        """One step of EDM sampler"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma
        
        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** 0.5 * epsilon if gamma > 0 else x

        # Evaluate ∂x/∂sigma at sigma_hat
        denoised_cur =  fn(x_hat, net=net, 
                           sigma=sigma_hat, inference=True, 
                           cond_scale=self.cond_scale, 
                           **kwargs)
        d = (x_hat - denoised_cur) / sigma_hat

        if sigma_next == 0.0:
            dt = sigma_next - sigma_hat
            x = x + d * dt
        else:
            # Denoise to midpoint
            sigma_mid = sigma_hat.log().lerp(sigma_next.log(), 0.5).exp()
            # sigma_mid = ((sigma_hat ** (1 / self.rho) + sigma_next ** (1 / self.rho)) / 2) ** self.rho
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigma_next - sigma_hat

            x_2 = x + d * dt_1

            denoised_2 = fn(x_2, net=net, 
                            sigma=sigma_mid, inference=True, 
                            cond_scale=self.cond_scale, 
                            **kwargs)
            d_2 = (x_2 - denoised_2) / sigma_mid
            x = x + d_2 * dt_2

        return x

    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs) -> Tensor:
        
        x = sigmas[0] * noise

        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / self.num_steps, sqrt(2) - 1),
            0.0,
        )
        
        # Denoise to sample
        for i in range(self.num_steps - 1):
            x = self.step(x, fn=fn, net=net, 
                          sigma=sigmas[i], 
                          sigma_next=sigmas[i+1], 
                          gamma=gammas[i],
                          **kwargs)

        return x.clamp(-1.0, 1.0)
    
class DPMSampler(nn.Module):

    """
    guided sampling with small guidance scale
    singlestep DPM-Solver ("DPM-Solver-fast" in the paper) with `order = 3`

    Deterministic sampler

    In EDM framework, corresponding variables are: 
    alpha = 1.0
    sigma = sigma
    lambd = log(alpha/sigma) = -log(sigma)

    single step: NFE <= num_steps // order + 1
    multistep: NFE = num_steps + 1
    
    """

    def __init__(self, cond_scale, 
                 order=1, 
                 num_steps=10, 
                 multisteps=False,
                 x0_pred: bool = True, 
                 log_time_spacing: bool = True):
        super().__init__()
        
        self.order = order
        self.cond_scale = cond_scale
        self.multisteps = multisteps
        self.x0_pred = x0_pred
        self.log_time_spacing = log_time_spacing
        self.num_steps = num_steps if self.log_time_spacing else num_steps - 1

    def lambd(self, sigma):
        if self.log_time_spacing:
            return sigma
        else:
            return -sigma.log()

    def sigma(self, lambd):
        if self.log_time_spacing:
            return lambd.neg().exp()
        else:
            return lambd

    def inv_lambd(self, lambd):
        if self.log_time_spacing:
            return lambd
        else:
            return lambd.neg().exp()

    def get_lambda(self, sigmas, num_steps):
        # logSNR space time spacing

        if self.log_time_spacing:
            lambd_start = -sigmas[0].log()
            lambd_end = -sigmas[-1].log()
            return torch.linspace(lambd_start, lambd_end, num_steps + 1)

        else:
            # the original time spacing
            return sigmas

    def eps(self, eps_cache, key, x, lambd, fn, 
            net, **kwargs):

        if key in eps_cache:
            return eps_cache[key], eps_cache

        eps = self.model_fn(x, fn, net, lambd, **kwargs)
        
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, lambd_cur, lambd_next, fn, net, eps_cache=None, **kwargs):

        h = self.lambd(lambd_next) - self.lambd(lambd_cur)
        eps_cache = {} if eps_cache is None else eps_cache
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

        if self.x0_pred:
            x_1 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - torch.expm1(-h) * eps

        else:
            x_1 = x - self.sigma(lambd_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, lambd_cur, lambd_next, fn, net, r1=1 / 2, eps_cache=None, **kwargs):

        h = self.lambd(lambd_next) - self.lambd(lambd_cur)
        s1 = lambd_cur + r1 * h
        s1 = self.inv_lambd(s1)
        eps_cache = {} if eps_cache is None else eps_cache
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

        if self.x0_pred:
            u1 = self.sigma(s1) / self.sigma(lambd_cur) * x - torch.expm1(-r1 * h) * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            x_2 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - torch.expm1(-h) * eps - 1 / (2 * r1) * torch.expm1(-h) * (eps_r1 - eps)

        else:
            u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            x_2 = x - self.sigma(lambd_next) * h.expm1() * eps - self.sigma(lambd_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, lambd_cur, lambd_next, fn, net, r1=1/3, r2=2/3, eps_cache=None, **kwargs):
        eps_cache = {} if eps_cache is None else eps_cache
        h = self.lambd(lambd_next) - self.lambd(lambd_cur)
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)
        s1 = lambd_cur + r1 * h
        s1 = self.inv_lambd(s1)
        s2 = lambd_cur + r2 * h
        s2 = self.inv_lambd(s2)

        if self.x0_pred:
            u1 = self.sigma(s1) / self.sigma(lambd_cur) * x - (-r1 * h).expm1() * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            u2 = self.sigma(s2) / self.sigma(lambd_cur) * x - (-r2 * h).expm1() * eps + (r2 / r1) * ((-r2 * h).expm1() / (r2 * h) + 1) * (eps_r1 - eps)
            eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2, fn, net)
            x_3 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - torch.expm1(-h) * eps + 1 / r2 * (torch.expm1(-h) / h + 1) * (eps_r2 - eps)

        else:
            u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
            eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2, fn, net)
            x_3 = x - self.sigma(lambd_next) * h.expm1() * eps - self.sigma(lambd_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache
    
    def multistep_dpm_solver_1_step(self, x, lambd_prev, lambd_cur, 
                                    fn, net, model_s=None, **kwargs):
        h = self.lambd(lambd_cur) - self.lambd(lambd_prev)

        if self.x0_pred:
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = fn(x, net=net, inference=True, 
                            sigma=self.sigma(lambd_prev), 
                            cond_scale=self.cond_scale, **kwargs)
            x_t =  self.sigma(lambd_cur) / self.sigma(lambd_prev) * x - phi_1 * model_s
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                net_pred = fn(x, net=net, inference=True, 
                            sigma=self.sigma(lambd_prev), 
                            cond_scale=self.cond_scale, **kwargs)
                model_s = (x - net_pred) / self.sigma(lambd_prev)
            x_t = x - self.sigma(lambd_cur) * phi_1 * model_s

        return x_t
    
    def multistep_dpm_solver_2_step(self, x, model_prev_list, lambd_prev_list, lambd_cur):

        lambd_prev_1, lambd_prev_0 = lambd_prev_list[-2], lambd_prev_list[-1]
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        h_1 = self.lambd(lambd_prev_0) - self.lambd(lambd_prev_1)
        h = self.lambd(lambd_cur) - self.lambd(lambd_prev_0)

        r0 = h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)

        if self.x0_pred:
            phi_1 = torch.expm1(-h)
            x_t = (self.sigma(lambd_cur) / self.sigma(lambd_prev_0) * x - phi_1 * model_prev_0 - 0.5 * phi_1 * D1_0)
        else:
            phi_1 = torch.expm1(h)
            x_t = (x - (self.sigma(lambd_cur) * phi_1) * model_prev_0 - 0.5 * (self.sigma(lambd_cur) * phi_1) * D1_0)
        return x_t
    
    def multistep_dpm_solver_3_step(self, x, model_prev_list, lambd_prev_list, lambd_cur):
        
        lambd_prev_2, lambd_prev_1, lambd_prev_0 = lambd_prev_list
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list

        h_1 = self.lambd(lambd_prev_1) - self.lambd(lambd_prev_2)
        h_0 = self.lambd(lambd_prev_0) - self.lambd(lambd_prev_1)
        h = self.lambd(lambd_cur) - self.lambd(lambd_prev_0)
        r0, r1 = h_0 / h, h_1 / h
        
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)

        if self.x0_pred:
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = self.sigma(lambd_cur) / self.sigma(lambd_prev_0) * x - phi_1 * model_prev_0 + phi_2 * D1 - phi_3 * D2

        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (x - (self.sigma(lambd_cur) * phi_1) * model_prev_0 - (self.sigma(lambd_cur) * phi_2) * D1 - (self.sigma(lambd_cur) * phi_3) * D2)
        return x_t

    def model_fn(self, x, fn, net, lambd, **kwargs):

        if self.x0_pred:
            model_t = fn(x, net=net, 
                        sigma=self.sigma(lambd), 
                        inference=True, 
                        cond_scale=self.cond_scale,
                        **kwargs)
        else:
            net_pred = fn(x, net=net, 
                        sigma=self.sigma(lambd), 
                        inference=True, 
                        cond_scale=self.cond_scale,
                        **kwargs)
            model_t = (x - net_pred) / self.sigma(lambd)

        return model_t
    
    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs):

        x = sigmas[0] * noise

        if self.multisteps:
            assert self.num_steps >= self.order

            lambds = self.get_lambda(sigmas, self.num_steps)

            # Init the initial values.
            step = 0
            lambd = lambds[step]
            model_t = self.model_fn(x, fn, net, lambd, **kwargs)
            model_prev_list = [model_t]
            lambd_prev_list = [lambd]

            for step in range(1, self.order):

                lambd_prev, lambd_cur = lambd_prev_list[-1], lambds[step]
                
                if step == 1:
                    x = self.multistep_dpm_solver_1_step(x, lambd_prev, lambd_cur, fn, net, 
                                                         model_s=model_prev_list[-1], **kwargs)
                elif step == 2:
                    x = self.multistep_dpm_solver_2_step(x, model_prev_list, lambd_prev_list, lambd_cur)
                elif step == 3:
                    x = self.multistep_dpm_solver_3_step(x, model_prev_list, lambd_prev_list, lambd_cur)
                
                lambd_prev_list.append(lambd_cur)
                model_t = self.model_fn(x, fn, net, lambd_cur, **kwargs)
                model_prev_list.append(model_t)

            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(self.order, self.num_steps + 1):

                lambd_prev, lambd_cur = lambd_prev_list[-1], lambds[step]
                
                step_order = min(self.order, self.num_steps + 1 - step) # We only use lower order for steps < 10

                if step_order == 1:
                    x = self.multistep_dpm_solver_1_step(x, lambd_prev, lambd_cur, fn, net, 
                                                         model_s=model_prev_list[-1], **kwargs)
                elif step_order == 2:
                    x = self.multistep_dpm_solver_2_step(x, model_prev_list, lambd_prev_list, lambd_cur)
                elif step_order == 3:
                    x = self.multistep_dpm_solver_3_step(x, model_prev_list, lambd_prev_list, lambd_cur)
                
                for i in range(self.order - 1):
                    lambd_prev_list[i] = lambd_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                lambd_prev_list[-1] = lambd_cur
                # We do not need to evaluate the final model value.
                if step < self.num_steps:
                    model_t = self.model_fn(x, fn, net, lambd_cur, **kwargs)
                    model_prev_list[-1] = model_t

        else:
            if self.order == 3:
                K = self.num_steps // 3 + 1
                if self.num_steps % 3 == 0:
                    orders = [3,] * (K - 2) + [2, 1]
                else:
                    orders = [3,] * (K - 1) + [self.num_steps % 3]

            elif self.order == 2:
                if self.num_steps % 2 == 0:
                    K = self.num_steps // 2
                    orders = [2,] * K
                else:
                    K = self.num_steps // 2 + 1
                    orders = [2,] * (K - 1) + [1]
            elif self.order == 1:
                K = self.num_steps
                orders = [1,] * self.num_steps
            else:
                raise ValueError("'order' must be '1' or '2' or '3'.")

            lambds = self.get_lambda(sigmas, K)

            for i in range(len(orders)):
                eps_cache = {}
                lambd_cur, lambd_next = lambds[i], lambds[i + 1]
                _, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

                if orders[i] == 1:
                    x, eps_cache = self.dpm_solver_1_step(x, lambd_cur, lambd_next, fn, net, eps_cache=eps_cache)
                elif orders[i] == 2:
                    x, eps_cache = self.dpm_solver_2_step(x, lambd_cur, lambd_next, fn, net, eps_cache=eps_cache)
                else:
                    x, eps_cache = self.dpm_solver_3_step(x, lambd_cur, lambd_next, fn, net, eps_cache=eps_cache)

        return x.clamp(-1.0, 1.0)

class UniPCSampler(nn.Module):
    
    """
    Uni-PC sampler, built upon multistep DPM
    we use x0 prediction method since it achieves better performance with EDM scheduler.

    NFE = num_steps
    
    reference: https://github.com/thu-ml/DPM-Solver-v3/blob/main/codebases/edm/samplers/uni_pc.py
    https://github.com/wl-zhao/UniPC
    """

    def __init__(self, num_steps: int = 20, 
                 order: int = 2,
                 cond_scale: float = 1.0,
                 x0_pred: bool = True, 
                 log_time_spacing: bool = True):
        
        super().__init__()
        
        self.order = order
        self.cond_scale = cond_scale
        self.x0_pred = x0_pred
        self.log_time_spacing = log_time_spacing
        self.num_steps = num_steps if self.log_time_spacing else num_steps - 1

    def model_fn(self, x, lambd, fn, net, **kwargs):
        
        if self.x0_pred:
            model_t = fn(x, net=net, 
                        sigma=self.lambda_to_sigma(lambd), 
                        inference=True, 
                        cond_scale=self.cond_scale, **kwargs)

        else:
            net_pred = fn(x, net=net, 
                          sigma=self.lambda_to_sigma(lambd), 
                          inference=True, 
                          cond_scale=self.cond_scale, **kwargs)
            model_t = (x - net_pred) / self.lambda_to_sigma(lambd)

        return model_t

    def sigma_to_lambd(self, sigma):
        if self.log_time_spacing:
            return sigma
        else:
            return -sigma.log()

    def lambda_to_sigma(self, lambd):
        if self.log_time_spacing:
            return lambd.neg().exp()
        else:
            return lambd

    def get_lambda(self, sigmas, num_steps):
        # logSNR space time spacing

        if self.log_time_spacing:
            lambd_start = -sigmas[0].log()
            lambd_end = -sigmas[-1].log()
            return torch.linspace(lambd_start, lambd_end, num_steps + 1)

        else:
            # the original time spacing
            return sigmas
    
    def multistep_uni_pc_update(self, x, model_prev_list, 
                                lambd_prev_list, lambd_cur, order, 
                                fn, net, x_t=None, use_corrector=True, 
                                variant='bh2', **kwargs):

        if len(lambd_cur.shape) == 0:
            lambd_cur = lambd_cur.view(-1)

        assert order <= len(model_prev_list)

        # first compute rks
        lambd_prev_0 = lambd_prev_list[-1]
        model_prev_0 = model_prev_list[-1]
        h = self.sigma_to_lambd(lambd_cur) - self.sigma_to_lambd(lambd_prev_0)

        rks = []
        D1s = []
        for i in range(1, order):

            lambd_prev_i = lambd_prev_list[-(i + 1)]
            rk = (self.sigma_to_lambd(lambd_prev_i)  - self.sigma_to_lambd(lambd_prev_0)) / h
            rks.append(rk)
            
            model_prev_i = model_prev_list[-(i + 1)]
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h if self.x0_pred else h
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if variant == 'bh1':
            B_h = hh
        elif variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
            
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        if self.x0_pred:

            x_t_ = self.lambda_to_sigma(lambd_cur) / self.lambda_to_sigma(lambd_prev_0) * x - h_phi_1 * model_prev_0

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, lambd_cur[0], fn, net, **kwargs)
                
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - B_h * (corr_res + rhos_c[-1] * D1_t)

        else:

            x_t_ = x - self.lambda_to_sigma(lambd_cur) * h_phi_1 * model_prev_0
            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0

                x_t = x_t_ - self.lambda_to_sigma(lambd_cur) * B_h * pred_res

            if use_corrector:

                model_t = self.model_fn(x_t, lambd_cur[0], fn, net, **kwargs)
                
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - self.lambda_to_sigma(lambd_cur) * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    @torch.no_grad()
    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs) -> Tensor:

        assert self.num_steps >= self.order
        x = sigmas[0] * noise
        lambds = self.get_lambda(sigmas, self.num_steps).to(x.device)

        # Init the initial values.
        step = 0
        lambd = lambds[step]
        model_t = self.model_fn(x, lambd, fn, net, **kwargs)

        model_prev_list = [model_t]
        lambd_prev_list = [lambd]

        for step in range(1, self.order):
            lambd_cur = lambds[step]
            x, model_x = self.multistep_uni_pc_update(x, model_prev_list, 
                                                      lambd_prev_list, lambd_cur, step, fn, 
                                                      net, use_corrector=True)
            if model_x is None:
                model_t = self.model_fn(x, lambd_cur, fn, net, **kwargs)

            lambd_prev_list.append(lambd_cur)
            model_prev_list.append(model_x)

        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(self.order, self.num_steps + 1):

            lambd_cur = lambds[step]

            step_order = min(self.order, self.num_steps + 1 - step)

            if step == self.num_steps:
                use_corrector = False
            else:
                use_corrector = True

            x, model_x = self.multistep_uni_pc_update(x, model_prev_list, 
                                                      lambd_prev_list, 
                                                      lambd_cur, 
                                                      step_order, fn, net, 
                                                      use_corrector=use_corrector)

            for i in range(self.order - 1):
                lambd_prev_list[i] = lambd_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            lambd_prev_list[-1] = lambd_cur
            # We do not need to evaluate the final model value.
            if step < self.num_steps:
                if model_x is None:
                    model_x = self.model_fn(x, lambd_cur, fn, net, **kwargs)

                model_prev_list[-1] = model_x

        return x.clamp(-1.0, 1.0)


class DPM2MSampler(nn.Module):

    """ 
    An implementation based on crowsonkb and hallatore from:
    https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457

    a.k.a DPM-Solver++(2M) Karras.

    Deterministic sampler
    """
    
    def __init__(self, num_steps: int = 50, cond_scale: float=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.cond_scale = cond_scale

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
                                    old_denoised=old_denoised,
                                    **kwargs)
            old_denoised = denoised

        return x.clamp(-1.0, 1.0)


class LMSSampler(nn.Module):
    '''
    Linear Multistep Solver.
    'LMS Karras', 'sample_lms'

    Deterministic sampler
    '''
    
    def __init__(self, num_steps: int = 50, cond_scale: float=1.0, order: int=4):
        super().__init__()

        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.order = order

    def linear_multistep_coeff(self, order, t, i, j):
        
        if order - 1 > i:
            raise ValueError(f'Order {order} too high for step {i}')
        def fn(tau):
            prod = 1.
            for k in range(order):
                if j == k:
                    continue
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
            return prod
        return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]
    
    def forward(self, noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, # actually t
                **kwargs) -> Tensor:
        
        sigmas_cpu = sigmas.detach().cpu().numpy()
        
        # Denoise to sample
        x = sigmas[0] * noise
        ds = []
        for i in range(self.num_steps - 1):
            x_epis = fn(x, net=net, sigma=sigmas[i], 
                        inference=True, 
                        cond_scale=self.cond_scale, 
                        **kwargs)
            d = (x - x_epis) / sigmas[i]
            ds.append(d)

            if len(ds) > self.order:
                ds.pop(0)

            cur_order = min(i + 1, self.order)

            coeffs = [self.linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x.clamp(-1.0, 1.0)
