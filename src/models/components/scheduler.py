import torch
import torch.nn as nn
from torch import Tensor
import math

class KarrasSchedule(nn.Module):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, 
                 rho: float = 7.0, num_steps: int = 50):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps

    def forward(self) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(self.num_steps, dtype=torch.float32)
        sigmas = (self.sigma_max ** rho_inv + steps / (self.num_steps - 1) * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)) ** self.rho

        return sigmas

class LinearSchedule(nn.Module):
    def __init__(self, start: float = 1.0, 
                 end: float = 0.0, 
                 num_steps: int = 50):
        super().__init__()
        self.start = start
        self.num_steps = num_steps
        self.end = end

    def forward(self) -> Tensor:

        sigmas = torch.linspace(self.start, self.end, self.num_steps)

        return sigmas
    
class GeometricSchedule(nn.Module):
    def __init__(self, sigma_max: float = 100, sigma_min: float = 0.02, num_steps: int = 50):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_steps = num_steps
        
    def forward(self) -> Tensor:
        
        steps = torch.arange(self.num_steps, dtype=torch.float32)
        sigmas = (self.sigma_max**2) * ((self.sigma_min**2/self.sigma_max**2) ** (steps / (self.num_steps-1)))
        
        return sigmas

class VPSchedule(nn.Module):
    def __init__(self, start: float = 1.0, 
                 end: float = 1e-3, 
                 beta_d: float = 19.9,
                 beta_min: float = 0.1,
                 num_steps: int = 50):
        super().__init__()
        
        self.start = start
        self.num_steps = num_steps
        self.end = end
        self.beta_d = beta_d
        self.beta_min = beta_min

    def forward(self) -> Tensor:
        
        sigmas = torch.linspace(self.start, self.end, self.num_steps)

        return ((0.5 * self.beta_d * (sigmas ** 2) + self.beta_min * sigmas).exp() - 1) ** 0.5

class VESchedule(nn.Module):
    def __init__(self, sigma_max: float = 100, sigma_min: float = 0.02, num_steps: int = 50):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_steps = num_steps
        
    def forward(self) -> Tensor:
        
        steps = torch.arange(self.num_steps, dtype=torch.float32)
        sigmas = (self.sigma_max**2) * ((self.sigma_min**2/self.sigma_max**2) ** (steps / (self.num_steps-1)))
        
        return sigmas.sqrt()

class VSchedule(nn.Module):
    def __init__(self, logsnr_min = -15, logsnr_max = 15, shift = 0.0, num_steps: int = 50):
        super().__init__()
        self.shift = shift
        self.num_steps = num_steps
        self.t_min = math.atan(math.exp(-0.5 * logsnr_max))
        self.t_max = math.atan(math.exp(-0.5 * logsnr_min))
        
    def forward(self) -> Tensor:
        
        t = torch.linspace(1.0, 0.0, self.num_steps)
        logsnr_t = -2 * (torch.tan(self.t_min + t * (self.t_max - self.t_min)).log()) + 2 * self.shift

        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))

        return sigma_t / alpha_t
    
class RFEDMSchedule(nn.Module):
    def __init__(self, start: float = 1.0, 
                 end: float = 0.0, 
                 num_steps: int = 50):
        super().__init__()

        assert start <= 1.0 and end >= 0.0, "start must be less than or equal to 1.0 and end must be greater than or equal to 0.0"
        self.start = start
        self.num_steps = num_steps
        self.end = end

    def forward(self) -> Tensor:

        sigmas = torch.linspace(self.start, self.end, self.num_steps)

        return sigmas / (1 - sigmas)