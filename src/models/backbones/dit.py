# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import numpy as np
from functools import partial
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from .utils import to_2tuple, default, exists
from .operator_utils import nchw_to, Format
from .attention_utils import Attention
try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message
from .conditioner import TimestepEmbedder, LabelEmbedder, TextEmbedder

#################################################################################
#                                   DiT Configs                                  
# DiT_XL_2: DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# DiT_XL_4: DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
        
# DiT_XL_8: DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# DiT_L_2: DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# DiT_L_4: DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# DiT_L_8: DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# DiT_B_2: DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# DiT_B_4: DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# DiT_B_8: DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# DiT_S_2: DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# DiT_S_4: DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# DiT_S_8: DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
        
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ViT layers
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: list = 224,
            patch_size: list = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, 
                 num_heads, mlp_ratio=4.0, 
                 use_self_text_cond=True,
                 use_qk_l2norm=False, use_rope=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim=hidden_size, 
                              heads=num_heads,
                              context_dim=hidden_size, 
                              use_self_text_cond=use_self_text_cond,
                              use_qk_l2norm=use_qk_l2norm,
                              use_rope=use_rope)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, 
                context=None, 
                context_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), 
                                                  context, context_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, np.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=[256, 128],
        patch_size=[8, 4],
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_drop_prob=0.1,
        num_classes=None,
        class_embed_dim=None,
        label_cond=False,
        text_cond=False,
        text_embed_dim=512,
        max_text_len=128,
        use_self_text_cond=True,
        use_qk_l2norm=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = list(patch_size)
        self.input_size = list(input_size)
        self.num_heads = num_heads
        self.cond_drop_prob = cond_drop_prob
        self.num_classes = num_classes
        self.label_cond = label_cond

        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, class_embed_dim, hidden_size, hidden_size) if label_cond else None
        self.text_conditioner = TextEmbedder(hidden_size, text_embed_dim, max_text_len) if text_cond else None

        num_patches = self.x_embedder.num_patches
        # Fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                     use_self_text_cond=use_self_text_cond,
                     use_qk_l2norm=use_qk_l2norm, use_rope=True) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        grid_size = tuple([s // p for s, p in zip(self.input_size, self.patch_size)])
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.label_cond:
            nn.init.normal_(self.y_embedder.label_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size[0]* patch_size[1] * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p1 = self.x_embedder.patch_size[0]
        p2 = self.x_embedder.patch_size[1]
        h = self.input_size[0] // p1
        w = self.input_size[1] // p2

        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p1, w * p2))
        return imgs

    def forward(self, 
                x: Tensor, 
                t: Tensor, 
                classes:Optional[Tensor] = None,         # class labels or class embeddings
                text_embeds:Optional[Tensor] = None,         # text embeddings
                text_mask:Optional[Tensor] = None,  
                cond_drop_prob=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, F, T) tensor of spatial inputs (images or latent representations of images)
        time: (N,) tensor of diffusion timesteps
        classes: (N,) tensor of class labels
        """
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        input_dim = len(x.shape)

        if input_dim == 3:
            x = x.unsqueeze(2)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)

        if exists(classes):
            c = self.y_embedder(classes, cond_drop_prob)    # (N, D) 
            c = c + t
        else:
            c = t

        # text condition
        if exists(text_embeds):
            context, text_mask = self.text_conditioner(text_embeds, text_mask, cond_drop_prob) 
        else:
            context = None
            text_mask = None
        

        for block in self.blocks:
            x = block(x, c, context, text_mask)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size[0] * patch_size[1] * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if input_dim == 3:
            x = x.squeeze(2)

        return x

