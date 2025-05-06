from math import pi
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
import copy
from .utils import exists, default
from .attention_utils import Attention
from .conditioner import TextEmbedder, LabelEmbedder

"""
Norms
"""

class LayerNorm(nn.Module):
    def __init__(self, features: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm


class LayerNorm1d(nn.Module):
    def __init__(self, channels: int, *, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.bias = bias
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channels, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        norm = (x - mean) * (var + self.eps).rsqrt() * self.g
        return norm + self.b if self.bias else norm

"""
Attention Helper Blocks
"""

def FeedForward1d(channels: int, multiplier: int = 2):
    mid_channels = int(channels * multiplier)
    return nn.Sequential(
        LayerNorm1d(channels=channels, bias=False),
        Conv1d(
            in_channels=channels, out_channels=mid_channels, kernel_size=1, bias=False
        ),
        nn.GELU(),
        LayerNorm1d(channels=mid_channels, bias=False),
        Conv1d(
            in_channels=mid_channels, out_channels=channels, kernel_size=1, bias=False
        ),
    )

"""
Transformer Blocks
"""

class TransformerBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        multiplier: int = 2,
        context_features: Optional[int] = None,
        use_self_text_cond: bool = False,
        use_qk_l2norm: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)
        self.use_self_text_cond = use_self_text_cond
        if context_features is not None:
            if not use_self_text_cond:
                self.attention = Attention(dim=channels, 
                                           heads=num_heads)
                self.cross_attention = Attention(dim=channels, 
                                                 heads=num_heads,
                                                 context_dim=context_features, 
                                                 use_self_text_cond=use_self_text_cond,
                                                 use_rope=use_rope)
                self.cross_norm = nn.LayerNorm(channels)
            else:
                self.attention = Attention(dim=channels, 
                                           heads=num_heads,
                                           context_dim=context_features, 
                                           use_self_text_cond=use_self_text_cond,
                                           use_qk_l2norm=use_qk_l2norm,
                                           use_rope=use_rope)

        else:
            self.attention = Attention(dim=channels, 
                                       heads=num_heads)
        
        self.feed_forward = FeedForward1d(channels=channels, multiplier=multiplier)

    def forward(self, x: Tensor, context = None, context_mask = None) -> Tensor:

        x = rearrange(x, 'b c l -> b l c')

        if context is None:
            x = self.attention(self.norm(x)) + x
        else:
            if not self.use_self_text_cond:
                x = self.attention(self.norm(x)) + x
                x = self.cross_attention(self.cross_norm(x), context, context_mask) + x
            else:
                x = self.attention(self.norm(x), context, context_mask) + x
        
        x = rearrange(x, 'b l c -> b c l')
        x = self.feed_forward(x) + x
        
        return x

"""
Time Embeddings
"""

class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )

"""
Convolutional Helper Blocks
"""

def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)

def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)

def scale_and_shift(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (scale + 1) + shift

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
        ) 

    def forward(
        self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None,
        inj_embeddings: Optional[Tensor] = None
    ) -> Tensor:
        
        x = self.groupnorm(x)
        if exists(scale_shift):
            x = scale_and_shift(x, scale=scale_shift[0], shift=scale_shift[1])
        
        if exists(inj_embeddings): 
            # inj_embeddings for diffae
            x = inj_embeddings * x

        x = self.activation(x)
        return self.project(x)


"""
UNet Helper Functions and Blocks
"""

def Downsample1d(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor*kernel_multiplier + 1,
        stride=factor,
        padding=factor*(kernel_multiplier//2),
    )

def Upsample1d(
    in_channels: int, out_channels: int, factor: int, use_nearest: bool = False
) -> nn.Module:

    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    if use_nearest:
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="nearest"),
            nn.ReflectionPad1d(1),
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
            ),
        )
    else:
        return ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
            output_padding=factor % 2,
        )

class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        dilation: int = 1,
        time_embed_dim: int = None, 
        classes_embed_dim: int = None,
    ) -> None:
        super().__init__()

        self.to_cond_embedding = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features=int(time_embed_dim or 0)+int(classes_embed_dim or 0),
                          out_features=out_channels*2
                ),
            ) if exists(time_embed_dim) or exists(classes_embed_dim) else None
        )

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            dilation=dilation,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            num_groups=num_groups,
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, 
                time_embed: Optional[Tensor] = None, 
                class_embed: Optional[Tensor] = None, 
                inj_embeddings: Optional[Tensor] = None) -> Tensor:

        # Compute scale and shift from conditional embedding (time_embed + class_embed)
        scale_shift = None
        if exists(self.to_cond_embedding) and (exists(time_embed) or exists(class_embed)):

            cond_emb = tuple(filter(exists, (time_embed, class_embed)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.to_cond_embedding(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x)
        h = self.block2(h, scale_shift=scale_shift, 
                        inj_embeddings=inj_embeddings)

        return h + self.to_out(x)

"""
UNet Blocks
"""

class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        use_attention: bool = False,
        time_embed_dim: int = None, 
        classes_embed_dim: int = None,
        attention_heads: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        text_embed_dim: Optional[int] = None,
        use_self_text_cond: Optional[bool] = False,
    ):
        super().__init__()

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_embed_dim=time_embed_dim, 
            classes_embed_dim=classes_embed_dim,
        )

        self.use_attention = use_attention
        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                multiplier=attention_multiplier,
                context_features=text_embed_dim,
                use_self_text_cond=use_self_text_cond,
            )
            
        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            time_embed_dim=time_embed_dim, 
            classes_embed_dim=classes_embed_dim,
        )

    def forward(self, x: Tensor, 
                t: Optional[Tensor] = None, 
                c: Optional[Tensor] = None, 
                context: Optional[Tensor] = None, 
                context_mask: Optional[Tensor] = None,
                inj_embeddings: Optional[Tensor] = None) -> Tensor:
        
        x = self.pre_block(x, t, c, inj_embeddings)

        if self.use_attention:
            x = self.transformer(x, context, context_mask)

        x = self.post_block(x, t, c, inj_embeddings)
        return x

class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        num_groups: int,
        num_layers: int,
        kernel_multiplier: int = 2,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        time_embed_dim: Optional[int]  = None, 
        classes_embed_dim: Optional[int] = None,
        text_embed_dim: Optional[int] = None,
        use_self_text_cond: Optional[bool] = False,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_attention = use_attention

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_embed_dim,
                )
                for i in range(num_layers)
            ]
        )
        
        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                multiplier=attention_multiplier,
                context_features=text_embed_dim,
                use_self_text_cond=use_self_text_cond,
            )

    def forward(
        self, x: Tensor, 
        t: Optional[Tensor] = None, 
        c: Optional[Tensor] = None, 
        context: Optional[Tensor] = None, 
        context_mask: Optional[Tensor] = None,
        inj_embeddings: Optional[Tensor] = None,
        inj_channels: Optional[Tensor] = None) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:
        
        if inj_channels is not None and inj_channels.shape[-1] == x.shape[-1]:
            x = x + inj_channels
        
        if self.use_pre_downsample:
            x = self.downsample(x)

        skips = []
        for block in self.blocks:
            x = block(x, t, c, inj_embeddings)
            skips += [x] if self.use_skip else []

        if self.use_attention:
            x = self.transformer(x, context, context_mask)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x)

        return (x, skips) if self.use_skip else x

class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_nearest: bool = False,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        use_attention: bool = False,
        attention_heads: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        time_embed_dim: int = None, 
        classes_embed_dim: int = None,
        text_embed_dim: Optional[int] = None,
        use_self_text_cond: Optional[bool] = False,
    ):
        super().__init__()

        assert (not use_attention) or (
            exists(attention_heads)
            and exists(attention_multiplier)
        )

        self.use_pre_upsample = use_pre_upsample
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_embed_dim,
                )
                for _ in range(num_layers)
            ]
        )

        if use_attention:
            assert (
                exists(attention_heads)
                and exists(attention_multiplier)
            )
            self.transformer = TransformerBlock1d(
                channels=channels,
                num_heads=attention_heads,
                multiplier=attention_multiplier,
                context_features=text_embed_dim,
                use_self_text_cond=use_self_text_cond,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
        self,
        x: Tensor,
        skips: Optional[List[Tensor]] = None,
        t: Optional[Tensor] = None,
        c: Optional[Tensor] = None, 
        context: Optional[Tensor] = None, 
        context_mask: Optional[Tensor] = None,
        inj_embeddings: Optional[Tensor] = None
    ) -> Tensor:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, t, c, inj_embeddings)

        if self.use_attention:
            x = self.transformer(x, context, context_mask)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        return x
    
"""
UNet and EncDec
"""
    
class WAVenc1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filters: int, 
        window_length: int, 
        stride: int,
    ):
        super().__init__()
        
        ## conv enc-dec LTP in https://github.com/archinetai/audio-diffusion-pytorch
        padding = window_length // 2 - stride // 2
        self.to_in = Conv1d(in_channels=in_channels,
                            out_channels=num_filters,
                            kernel_size=window_length,
                            stride=stride,
                            padding=padding,
                            bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.to_in(x)
        
class WAVdec1d(nn.Module):
    # efficient input transform for UNet:
    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        window_length: int, 
        stride: int,
        # output channels
        out_channels: Optional[int] = None
    ):

        super().__init__()
        
        out_channels = default(out_channels, in_channels)
        
        ## conv enc-dec LTP in https://github.com/archinetai/audio-diffusion-pytorch
        padding = window_length // 2 - stride // 2
        self.to_out = nn.ConvTranspose1d(in_channels=num_filters,
                                         out_channels=out_channels, 
                                         kernel_size=window_length,
                                         stride=stride, 
                                         padding=padding, 
                                         bias=False)

        # init output layer weights to zero, so that the expected value of the output is zero
        nn.init.constant_(self.to_out.weight, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.to_out(x)
    
class UNet1d(nn.Module):
    def __init__(
        self,
        # efficient input transform:
        num_filters: int,
        window_length: int, 
        stride: int,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[bool],
        attention_heads: int,
        attention_multiplier: int,
        resnet_groups: int,
        kernel_multiplier_downsample: int,
        use_nearest_upsample: bool,
        use_skip_scale: bool,
        use_attention_bottleneck: bool,
        use_condition_block: bool = False,
        out_channels: Optional[int] = None,
        classes_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        use_self_text_cond: Optional[bool] = False,
    ):
        super().__init__()

        self.factors = factors
        
        # outmost (input and output) blocks 
        self.to_in = WAVenc1d(in_channels=in_channels, 
                              num_filters=num_filters, 
                              window_length=window_length, 
                              stride=stride)

        self.to_out = WAVdec1d(in_channels=in_channels, 
                               num_filters=num_filters, 
                               window_length=window_length, 
                               stride=stride,
                               out_channels=out_channels)
        
        # dimension assertion
        time_embed_dim = channels * 4
        
        num_layers = len(multipliers) - 1
        self.num_layers = num_layers
        
        assert (len(factors) == num_layers
                and len(attentions) == num_layers
                and len(num_blocks) == num_layers
               )

        # time embedding
        self.to_time = nn.Sequential(
            TimePositionalEmbedding(dim=channels, out_features=time_embed_dim),
            nn.SiLU(),
            nn.Linear(
                in_features=time_embed_dim, out_features=time_embed_dim
            ),
        )

        # unet
        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_dim,
                    text_embed_dim=text_dim,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_multiplier=attention_multiplier,
                    use_self_text_cond=use_self_text_cond,
                )
                for i in range(num_layers)
            ]
        )
        
        if use_condition_block:
            self.condition_to_in = copy.deepcopy(self.to_in)
            self.condition_block = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i], 
                    out_channels=channels * multipliers[i + 1],
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_dim,
                    text_embed_dim=text_dim,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=False,
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_multiplier=attention_multiplier
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            num_groups=resnet_groups,
            use_attention=use_attention_bottleneck,
            attention_heads=attention_heads,
            attention_multiplier=attention_multiplier,
            time_embed_dim=time_embed_dim, 
            classes_embed_dim=classes_dim,
            text_embed_dim=text_dim,
            use_self_text_cond=use_self_text_cond,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    time_embed_dim=time_embed_dim, 
                    classes_embed_dim=classes_dim,
                    text_embed_dim=text_dim,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    skip_channels=channels * multipliers[i + 1],
                    use_attention=attentions[i],
                    attention_heads=attention_heads,
                    attention_multiplier=attention_multiplier,
                    use_self_text_cond=use_self_text_cond,
                )
                for i in reversed(range(num_layers))
            ]
        )

    def forward(self, 
                x: Tensor, 
                t: Tensor, 
                classes: Tensor, 
                context:Optional[Tensor] = None, 
                inj_embeddings:Optional[Tensor] = None, 
                inj_channels:Optional[Tensor] = None, 
                context_mask:Optional[Tensor] = None, 
                **kwargs):

        # input transform
        x = self.to_in(x)

        # inject channel type condition
        if inj_channels is not None:
            inj_channels = self.condition_to_in(inj_channels)

        # unet
        # time embedding
        t = self.to_time(t)
        skips_list = []
        for i, downsample in enumerate(self.downsamples):
            x, skips = downsample(x, t, classes,
                                  context=context,
                                  context_mask=context_mask,
                                  inj_embeddings=inj_embeddings,
                                  inj_channels=inj_channels) 
                
            if inj_channels is not None:
                inj_channels = self.condition_block[i](inj_channels)             
            skips_list += [skips]

        x = self.bottleneck(x, t, classes, context=context, 
                            context_mask=context_mask,
                            inj_embeddings=inj_embeddings)
        
        for _, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips, t, classes, 
                         context=context, 
                         context_mask=context_mask,
                         inj_embeddings=inj_embeddings)
            
        # output transform
        x = self.to_out(x) 
        return x
    
class UNet1dBase(nn.Module):
    # unet1d based diffusion with CFG
    
    def __init__(self, 
                 channels: int, 
                 cond_drop_prob: float, 
                 num_classes: int = None,
                 class_embed_dim: int = None,
                 class_cond: bool = False,
                 text_cond: bool = False,
                 max_text_len: int = None,
                 text_embed_dim = 768, # Bert embedding dim
                 text_cond_multiplier: int = None,
                 use_self_text_cond: bool = False,
                 use_condition_block: bool = False,
                 # conditional dropout of the time, must be greater than 0. to unlock classifier free guidance
                 **kwargs):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob  # classifier free guidance dropout

        # class embeddings for cfg
        classes_channels = None
        if class_cond:
            classes_channels = channels * 4
            assert num_classes is not None or class_embed_dim is not None
            self.label_conditioner = LabelEmbedder(num_classes, 
                                                   class_embed_dim, 
                                                   channels, 
                                                   classes_channels)
            
        # text embeddings for cfg     
        # TODO: add pooled text embedding   
        text_cond_dim = None
        if text_cond:
            text_cond_multiplier = default(text_cond_multiplier, 4)
            text_cond_dim = channels * text_cond_multiplier
            self.text_conditioner = TextEmbedder(text_cond_dim, text_embed_dim, max_text_len)
            
        self.unet = UNet1d(channels=channels, 
                           classes_dim=classes_channels,
                           text_dim=text_cond_dim,
                           use_self_text_cond=use_self_text_cond,
                           use_condition_block=use_condition_block,
                           **kwargs)

    def forward(self, 
                x: Tensor, 
                t: Tensor, 
                classes:Optional[Tensor] = None,         # class labels or class embeddings
                text_embeds:Optional[Tensor] = None,         # text embeddings
                text_mask:Optional[Tensor] = None,  
                inj_embeddings:Optional[Tensor] = None,  # diffae
                inj_channels:Optional[Tensor] = None,    # channel type condition
                cond_drop_prob=None, **kwargs):

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        
        # label condition
        classes_emb = self.label_conditioner(classes, cond_drop_prob) if exists(classes) else None

        # text condition
        if exists(text_embeds):
            context, text_mask = self.text_conditioner(text_embeds, text_mask, cond_drop_prob) 
        else:
            context = None
            text_mask = None

        # unet
        x = self.unet(x, t, classes=classes_emb, 
                      context=context, 
                      context_mask=text_mask,
                      inj_embeddings=inj_embeddings, 
                      inj_channels=inj_channels, **kwargs)

        return x

# example

if __name__ == '__main__':
    num_classes = 0

    model = UNet1dBase(
        text_cond=True,
        text_embed_dim=768,
        max_text_len=10,
        num_classes= 0,
        cond_drop_prob= 0.1,
        num_filters=128,
        window_length=3, 
        stride=1,
        channels=128,
        in_channels=32,
        resnet_groups=8,
        kernel_multiplier_downsample=2,
        multipliers=[1,2,4,4,4,4,4],
        factors=[1,1,1,2,2,2],
        num_blocks=[2,2,2,2,2,2],
        attentions=[False,False,False,True,True,True],
        attention_heads=8,
        attention_multiplier=2,
        use_condition_block=True,
        use_nearest_upsample=True,
        use_skip_scale=True,
        use_attention_bottleneck=True,
        )
    

    training_audios = torch.randn(1, 32, 368) # images are normalized from 0 to 1
    text_bert = torch.randn(1, 10, 768)

    x_hat = model(training_audios, torch.tensor([0.5]), None, text_bert, text_mask=None)
    print(x_hat.shape)
#     torchaudio.save('./demo.wav', x_hat[0], 44100