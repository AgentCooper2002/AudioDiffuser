import torch.nn as nn
import torch
from torch import Tensor
from .dac.layers import WNConv1d, Snake1d

def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
    losses = mean**2 + logvar.exp() - logvar - 1
    loss = 0.5 * torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)
    return loss

class FineTuneAutoencoder(nn.Module):

    def __init__(self, intermediate_embedding_size=[1024, 512, 256, 128],
                 conv_kernel=3, 
                 tanh_btnk=False,
                 latent_dim=32):
        super(FineTuneAutoencoder, self).__init__()
        
        assert intermediate_embedding_size[0] == 1024 # original DAC embedding size

        self.encoder_layers = []
        self.decoder_layers = []

        # VAE bottleneck
        self.btnk_layer = nn.Conv1d(intermediate_embedding_size[-1], 
                                    latent_dim * 2, 
                                    kernel_size=1, 
                                    stride=1)
        self.tanh_btnk = tanh_btnk

        # VAE encoder
        for input_ch, output_ch in zip(intermediate_embedding_size[:-1], intermediate_embedding_size[1:]):
            self.encoder_layers.append(Snake1d(input_ch))
            self.encoder_layers.append(WNConv1d(input_ch, output_ch, kernel_size=conv_kernel, stride=1, padding=1))

        # VAE decoder
        self.decoder_layers.append(WNConv1d(latent_dim, intermediate_embedding_size[-1], 
                                            kernel_size=conv_kernel, stride=1, padding=1))
        for input_ch, output_ch in zip(intermediate_embedding_size[::-1][:-1], intermediate_embedding_size[::-1][1:]):
            self.decoder_layers.append(Snake1d(input_ch))
            self.decoder_layers.append(WNConv1d(input_ch, output_ch, kernel_size=conv_kernel, stride=1, padding=1))

        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

    def vae_sample(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def encode(self, x, is_train=True):

        # encoder layer 
        x = self.encoder_layers(x)

        # btnk layer with mean and var
        mean_and_logvar = self.btnk_layer(x)
        mean, logvar = mean_and_logvar.chunk(chunks=2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        if self.tanh_btnk:
            mean = torch.tanh(mean)
        kl_loss_value = kl_loss(mean, logvar) # vae loss

        # sample from mean and var
        if is_train:
            return self.vae_sample(mean, logvar), kl_loss_value
        else:
            return mean, kl_loss_value

    def decode(self, x):
        x = self.decoder_layers(x)
        return x

    def forward(self, x, is_train=True):
        
        x, kl_loss_value = self.encode(x, is_train)
        x = self.decode(x)
        return x, kl_loss_value
