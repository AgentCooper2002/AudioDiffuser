from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
import math
from torch.special import expm1

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

class VEulerSampler(nn.Module):
    """ 
    V-diffusion Sampler from simple diffusion, using shifted noise scheduler, deterministic sampler

    """
    def __init__(
        self,
        logsnr_min = -15, 
        logsnr_max = 15,
        shift = 0.5,
        num_steps = 200,
        cond_scale = 1.0,
        use_heun = False,
    ):
        super().__init__()
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.shift = shift
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.use_heun = use_heun
    
    def step(self, 
             x: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             t: float, 
             t_next: float, 
             **kwargs) -> Tensor:
        
        logsnr_t = self.shifted_cosine_transform(t)
        logsnr_s = self.shifted_cosine_transform(t_next)

        # One step v-prediction
        v_pred = fn(x, net=net, sigma=logsnr_t, inference=True, cond_scale=self.cond_scale, **kwargs)
        
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))

        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        if t_next == 0.0:
            x_next = alpha_t * x - sigma_t * v_pred
        else:
            score_cur = - alpha_t * sigma_t * v_pred
            x_next = x + 0.5 * (logsnr_s - logsnr_t) * score_cur

            if self.use_heun:
                # Second step v-prediction
                v_pred_next = fn(x_next, net=net, sigma=logsnr_s, inference=True, cond_scale=self.cond_scale, **kwargs)
                score_next = - alpha_s * sigma_s * v_pred_next
                x_next = x + 0.25 * (logsnr_s - logsnr_t) * (score_next + score_cur)
            
        return x_next #.clamp(-1.0, 1.0)

    def shifted_cosine_transform(self, t: Tensor) -> Tensor:
        t_min = math.atan(math.exp(-0.5 * self.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.logsnr_min))
        return -2 * (torch.tan(t_min + t * (t_max - t_min)).log()) + 2 * self.shift
    
    def forward(self, 
                noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, # actually t_raw
                **kwargs) -> Tensor:

        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])]) # t_N = 0

        # Denoise to sample
        x = noise
        for i in range(self.num_steps):
            x = self.step(x, fn=fn,
                          net=net,
                          t=sigmas[i],
                          t_next=sigmas[i+1],
                          **kwargs)
        return x.clamp(-1.0, 1.0)

class VSampler(nn.Module):
    """ 
    V-diffusion Sampler from simple diffusion, using shifted noise scheduler, stochastic sampler

    """
    def __init__(
        self,
        logsnr_min = -15, 
        logsnr_max = 15,
        shift = 0.0,
        num_steps = 200,
        cond_scale = 1.0,
    ):
        super().__init__()
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.shift = shift
        self.num_steps = num_steps
        self.cond_scale = cond_scale
    
    def step(self, 
             x: Tensor, 
             fn: Callable, 
             net: nn.Module, 
             t: float, 
             t_next: float, 
             **kwargs) -> Tensor:
        
        logsnr_t = self.shifted_cosine_transform(t)
        logsnr_s = self.shifted_cosine_transform(t_next)

        # One step v-prediction
        v_pred = fn(x, net=net, sigma=logsnr_t, inference=True, cond_scale=self.cond_scale, **kwargs)
        
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))

        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        x_pred = alpha_t * x - sigma_t * v_pred
        x_pred = x_pred.clamp(-1.0, 1.0)

        c = -expm1(logsnr_t - logsnr_s)
        mu = alpha_s * (x * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c
        
        if t_next != 0:
            x_pred = mu + torch.randn_like(mu) * torch.sqrt(variance)
        else:
            x_pred = mu

        # if t_next != 0:
        #     eps_pred = sigma_t * x + alpha_t * v_pred
        #     x_next = sigma_s * eps_pred + alpha_s * x0_pred     # deterministic
        # else:
        #     x_next = x0_pred
            
        return x_pred

    def shifted_cosine_transform(self, t: Tensor) -> Tensor:
        t_min = math.atan(math.exp(-0.5 * self.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.logsnr_min))
        return -2 * (torch.tan(t_min + t * (t_max - t_min)).log()) + 2 * self.shift
    
    def forward(self, 
                noise: Tensor, 
                fn: Callable, 
                net: nn.Module, 
                sigmas: Tensor, # actually t_raw
                **kwargs) -> Tensor:
        
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])]) # t_N = 0


        # Denoise to sample
        x = noise
        for i in range(self.num_steps):
            x = self.step(x, fn=fn,
                          net=net,
                          t=sigmas[i],
                          t_next=sigmas[i+1],
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
                 x0_pred: bool = True):
        super().__init__()
        
        self.order = order
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.multisteps = multisteps
        self.x0_pred = x0_pred

    def lambd(self, sigma):
        shift = 0.0
        logsnr_min = -15 
        logsnr_max = 15
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * (torch.tan(t_min + sigma * (t_max - t_min)).log()) + 2 * shift

    def sigma(self, lambd):
        return torch.sqrt(torch.sigmoid(-lambd))

    def eps(self, eps_cache, key, x, lambd, fn, 
            net, **kwargs):

        if key in eps_cache:
            return eps_cache[key], eps_cache

        eps = self.model_fn(x, lambd, fn, net, **kwargs)

        return eps, {key: eps, **eps_cache}

    def model_fn(self, x, lambd, fn, net, **kwargs):

        v_pred = fn(x, net=net, 
                    sigma=lambd, 
                    inference=True, 
                    cond_scale=self.cond_scale,
                    **kwargs)
            
        if self.x0_pred:
            model_t = self.sigma(-lambd) * x - self.sigma(lambd) * v_pred
        else:
            model_t = self.sigma(lambd) * x + self.sigma(-lambd) * v_pred
    
        return model_t

    def dpm_solver_1_step(self, x, lambd_cur, lambd_next, fn, net, eps_cache=None, **kwargs):

        h = lambd_next - lambd_cur
        h = h / 2
        eps_cache = {} if eps_cache is None else eps_cache
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

        if self.x0_pred:
            x_1 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - self.sigma(-lambd_next) * (-h).expm1() * eps
        else:
            x_1 = self.sigma(-lambd_next) / self.sigma(-lambd_cur) * x - self.sigma(lambd_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, lambd_cur, lambd_next, fn, net, r1=1 / 2, eps_cache=None, **kwargs):

        h = lambd_next - lambd_cur
        s1 = lambd_cur + r1 * h
        h = h/2
        eps_cache = {} if eps_cache is None else eps_cache
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

        if self.x0_pred:
            u1 = self.sigma(s1) / self.sigma(lambd_cur) * x - self.sigma(-s1) * torch.expm1(-r1 * h) * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            x_2 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - self.sigma(-lambd_next) * torch.expm1(-h) * eps - self.sigma(-lambd_next) / (2 * r1) * torch.expm1(-h) * (eps_r1 - eps)

        else:
            u1 = self.sigma(-s1) / self.sigma(-lambd_cur) * x - self.sigma(s1) * (r1 * h).expm1() * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            x_2 = self.sigma(-lambd_next) / self.sigma(-lambd_cur) * x - self.sigma(lambd_next) * (h).expm1() * eps - self.sigma(lambd_next) / (2 * r1) * (h).expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, lambd_cur, lambd_next, fn, net, r1=1/3, r2=2/3, eps_cache=None, **kwargs):
        eps_cache = {} if eps_cache is None else eps_cache
        h = lambd_next - lambd_cur
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)
        s1 = lambd_cur + r1 * h
        s2 = lambd_cur + r2 * h
        h = h / 2

        if self.x0_pred:
            u1 = self.sigma(s1) / self.sigma(lambd_cur) * x - self.sigma(-s1) * (-r1 * h).expm1() * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            u2 = self.sigma(s2) / self.sigma(lambd_cur) * x - self.sigma(-s2) * (-r2 * h).expm1() * eps + self.sigma(-s2) * (r2 / r1) * ((-r2 * h).expm1() / (r2 * h) + 1) * (eps_r1 - eps)
            eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2, fn, net)
            x_3 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - self.sigma(-lambd_next) * torch.expm1(-h) * eps + self.sigma(-lambd_next) * 1 / r2 * (torch.expm1(-h) / h + 1) * (eps_r2 - eps)

        else:
            u1 = self.sigma(-s1) / self.sigma(-lambd_cur) * x - self.sigma(s1) * (r1 * h).expm1() * eps
            eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
            u2 = self.sigma(-s2) / self.sigma(-lambd_cur) * x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
            eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2, fn, net)
            x_3 = self.sigma(-lambd_next) / self.sigma(-lambd_cur) * x - self.sigma(lambd_next) * h.expm1() * eps - self.sigma(lambd_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def multistep_dpm_solver_1_step(self, x, lambd_prev, lambd_cur, 
                                    fn, net, model_s=None, **kwargs):
        h = lambd_cur - lambd_prev
        h = h / 2

        v_pred = fn(x, net=net, inference=True, 
                    sigma=lambd_prev, 
                    cond_scale=self.cond_scale, **kwargs)

        if self.x0_pred:
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s =  self.sigma(-lambd_prev) * x - self.sigma(lambd_prev) * v_pred
            x_t =  self.sigma(lambd_cur) / self.sigma(lambd_prev) * x - self.sigma(-lambd_cur) * phi_1 * model_s
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.sigma(lambd_prev) * x + self.sigma(-lambd_prev) * v_pred
            x_t = self.sigma(-lambd_cur) / self.sigma(-lambd_prev) * x - self.sigma(lambd_cur) * phi_1 * model_s

        return x_t

    def multistep_dpm_solver_2_step(self, x, model_prev_list, lambd_prev_list, lambd_cur):

        lambd_prev_1, lambd_prev_0 = lambd_prev_list[-2], lambd_prev_list[-1]
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        h_1 = lambd_prev_0 - lambd_prev_1
        h = lambd_cur - lambd_prev_0
        r0 = h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        h = h / 2

        if self.x0_pred:
            phi_1 = torch.expm1(-h)
            x_t = (self.sigma(lambd_cur) / self.sigma(lambd_prev_0) * x - self.sigma(-lambd_cur) * phi_1 * model_prev_0 - self.sigma(-lambd_cur) * 0.5 * phi_1 * D1_0)
        else:
            phi_1 = torch.expm1(h)
            x_t = (self.sigma(-lambd_cur) / self.sigma(-lambd_prev_0) * x - (self.sigma(lambd_cur) * phi_1) * model_prev_0 - 0.5 * (self.sigma(lambd_cur) * phi_1) * D1_0)
        return x_t

    def multistep_dpm_solver_3_step(self, x, model_prev_list, lambd_prev_list, lambd_cur):
        
        lambd_prev_2, lambd_prev_1, lambd_prev_0 = lambd_prev_list
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list

        h_1 = lambd_prev_1 - lambd_prev_2
        h_0 = lambd_prev_0 - lambd_prev_1
        h = lambd_cur - lambd_prev_0
        r0, r1 = h_0 / h, h_1 / h
        h = h / 2
        
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        
        if self.x0_pred:
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = self.sigma(lambd_cur) / self.sigma(lambd_prev_0) * x - self.sigma(-lambd_cur) * phi_1 * model_prev_0 + self.sigma(-lambd_cur) * phi_2 * D1 - self.sigma(-lambd_cur) * phi_3 * D2

        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (self.sigma(-lambd_cur) / self.sigma(-lambd_prev_0) * x - (self.sigma(lambd_cur) * phi_1) * model_prev_0 - (self.sigma(lambd_cur) * phi_2) * D1 - (self.sigma(lambd_cur) * phi_3) * D2)
        return x_t

    
    
    def forward(self, noise: Tensor, 
                fn: Callable,  # denoising function
                net: nn.Module, 
                sigmas: Tensor, 
                **kwargs):

        x = noise

        lambd_start = self.lambd(sigmas[0])
        lambd_end = self.lambd(sigmas[-1])

        if self.multisteps:
            assert self.num_steps >= self.order

            lambds = torch.linspace(lambd_start, lambd_end, self.num_steps + 1, device=x.device)

            # Init the initial values.
            step = 0
            lambd = lambds[step]

            v_pred = fn(x, net=net, 
                        sigma=lambd, 
                        inference=True, 
                        cond_scale=self.cond_scale,
                        **kwargs)
            
            if self.x0_pred:
                model_t = self.sigma(-lambd) * x - self.sigma(lambd) * v_pred
            else:
                model_t = self.sigma(lambd) * x + self.sigma(-lambd) * v_pred

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
                model_t = self.model_fn(x, lambd_cur, fn, net, **kwargs)
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
                    model_t = self.model_fn(x, lambd_cur, fn, net, **kwargs)
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
            
            lambds = torch.linspace(lambd_start, lambd_end, K + 1, device=x.device)

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
                 x0_pred: bool = True):
        
        super().__init__()
        self.num_steps = num_steps
        self.order = order
        self.cond_scale = cond_scale
        self.x0_pred = x0_pred

    def model_fn(self, x, lambd, fn, net, **kwargs):

        v_pred = fn(x, net=net, 
                    sigma=lambd, 
                    inference=True, 
                    cond_scale=self.cond_scale,
                    **kwargs)

        if self.x0_pred:
            model_t = self.lambda_to_sigma(-lambd) * x - self.lambda_to_sigma(lambd) * v_pred
        else:
            model_t = self.lambda_to_sigma(lambd) * x + self.lambda_to_sigma(-lambd) * v_pred

        return model_t

    def sigma_to_lambd(self, sigma):
        shift = 0.0
        logsnr_min = -15 
        logsnr_max = 15
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * (torch.tan(t_min + sigma * (t_max - t_min)).log()) + 2 * shift

    def lambda_to_sigma(self, lambd):
        return torch.sqrt(torch.sigmoid(-lambd))
    
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
        h = lambd_cur - lambd_prev_0

        rks = []
        D1s = []
        for i in range(1, order):

            lambd_prev_i = lambd_prev_list[-(i + 1)]
            rk = (lambd_prev_i  - lambd_prev_0) / h
            rks.append(rk)
            
            model_prev_i = model_prev_list[-(i + 1)]
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h/2 if self.x0_pred else h/2
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
            x_t_ = self.lambda_to_sigma(lambd_cur) / self.lambda_to_sigma(lambd_prev_0) * x - self.lambda_to_sigma(-lambd_cur) * h_phi_1 * model_prev_0

            if x_t is None:
                if use_predictor:
                    pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
                else:
                    pred_res = 0
                x_t = x_t_ - self.lambda_to_sigma(-lambd_cur) * B_h * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, lambd_cur[0], fn, net, **kwargs)
                
                if D1s is not None:
                    corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
                else:
                    corr_res = 0
                D1_t = (model_t - model_prev_0)
                x_t = x_t_ - B_h * (corr_res + rhos_c[-1] * D1_t)

        else:

            x_t_ = self.lambda_to_sigma(-lambd_cur) / self.lambda_to_sigma(-lambd_prev_0) * x - self.lambda_to_sigma(lambd_cur) * h_phi_1 * model_prev_0
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
    
        lambd_start = self.sigma_to_lambd(sigmas[0])
        lambd_end = self.sigma_to_lambd(sigmas[-1])
        lambds = torch.linspace(lambd_start, lambd_end, self.num_steps + 1, device=x.device)

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