import math
from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
import copy
from .utils import exists, default, cast_tuple
from .operator_utils import prob_mask_like, resize_image_to, Identity, Always, Parallel
from .attention_utils import Attention, LayerNorm, FeedForward, LinearAttention, ChanFeedForward
from .layer_utils import cond_weight_norm
from .conditioner import LabelEmbedder, TextEmbedder

# decoder
def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x).contiguous()

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x).contiguous()

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        linear_attn = False,
        use_gca = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = Attention if not linear_attn else LinearAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()


    def forward(self, x, time_emb=None, cond=None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)

        if cond is not None and exists(self.cross_attn):
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context=cond) + h
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        h = self.block2(h, scale_shift=scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x).contiguous()

class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth = 1,
        heads = 8,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.norm = LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, 
                          heads=heads, 
                          context_dim=context_dim),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context=None, context_mask=None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            if context is None:
                x = attn(self.norm(x)) + x
            else:
                x = attn(self.norm(x), context, context_mask) + x
            x = ff(x).contiguous() + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        ff_mult = 2,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, context_dim = context_dim),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, 
                                        stride=stride, padding=(kernel-stride)//2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x).contiguous(), self.convs))
        return torch.cat(fmaps, dim = 1)

class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)
    
class DownsamplingBlock(nn.Module):
    def __init__(self, dim_in, dim_out, 
                 cond_dim: int,
                 time_cond_dim: int,
                 attn_heads: int,
                 use_global_context_attn: bool,
                 layer_num_resnet_blocks, 
                 groups: int, 
                 layer_attn: bool, 
                 layer_attn_depth: int, 
                 layer_cross_attn: bool, 
                 layer_use_linear_attn: bool, 
                 layer_use_linear_cross_attn: bool, 
                 ff_mult: float,
                 memory_efficient: bool, 
                 is_last_layer: bool, 
                 cross_embed_downsample: bool,
                 cross_embed_downsample_kernel_sizes: tuple, 
                 channel_infuse_mode: str):
        
        super().__init__()
        self.channel_infuse_mode = channel_infuse_mode
        if channel_infuse_mode == 'conv':
            self.channel_cond_conv = cond_weight_norm(nn.Conv2d(dim_in*2, dim_in, 3, padding=1))

        layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

        # resnet block klass
        attn_kwargs = dict(heads=attn_heads)
        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        # downsample klass define
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, 
                                       kernel_sizes=cross_embed_downsample_kernel_sizes)
        else:
            downsample_klass = Downsample

        # entry attention block
        if layer_attn:
            transformer_block_klass = TransformerBlock
        elif layer_use_linear_attn:
            transformer_block_klass = LinearAttentionTransformerBlock
        else:
            transformer_block_klass = Identity

        if memory_efficient:
            # whether to pre-downsample, from memory efficient unet
            pre_downsample = downsample_klass(dim_in, dim_out)
            current_dim = dim_out
            post_downsample = None
            current_dim = dim_out
        else:
            # whether to do post-downsample, for non-memory efficient unet
            pre_downsample = None
            current_dim = dim_in

            if not is_last_layer:
                post_downsample = downsample_klass(current_dim, dim_out)  
            else: 
                post_downsample = Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), 
                                            nn.Conv2d(dim_in, dim_out, 1))
        
        self.ds_block = nn.ModuleList([pre_downsample,
                                       resnet_klass(current_dim, current_dim, 
                                                    cond_dim = layer_cond_dim, 
                                                    linear_attn = layer_use_linear_cross_attn, 
                                                    time_cond_dim = time_cond_dim, groups = groups),
                                                    
                                        nn.ModuleList([ResnetBlock(current_dim, current_dim, 
                                                                   time_cond_dim = time_cond_dim, 
                                                                   groups = groups, 
                                                                   use_gca = use_global_context_attn
                                                                   ) for _ in range(layer_num_resnet_blocks)]),
                                        transformer_block_klass(dim = current_dim, 
                                                                depth = layer_attn_depth, 
                                                                ff_mult = ff_mult, 
                                                                context_dim = cond_dim, 
                                                                **attn_kwargs),
                                        post_downsample])
        

    def forward(self, x, 
                t=None, c=None, 
                context=None, 
                context_mask=None,
                hiddens=None, 
                inj_channels=None):
        
        if inj_channels is not None:
            if self.channel_infuse_mode == 'conv':
                x = self.channel_cond_conv(torch.cat((x, inj_channels), dim=1))
            elif self.channel_infuse_mode == 'add':
                x = (x + inj_channels) / math.sqrt(2)

        pre_downsample, init_block, resnet_blocks, attn_block, post_downsample = self.ds_block
        
        if exists(pre_downsample):
            x = pre_downsample(x)

        x = init_block(x, time_emb=t, cond=c)

        for resnet_block in resnet_blocks:
            x = resnet_block(x, time_emb=t)
            if hiddens is not None:
                hiddens.append(x)

        x = attn_block(x, context=context, context_mask=context_mask)
        if hiddens is not None:
            hiddens.append(x)

        if exists(post_downsample):
            x = post_downsample(x)
        
        return x, hiddens

class MiddleBlock(nn.Module):

    def __init__(self, mid_dim, cond_dim, 
                 time_cond_dim, mid_resnet_group, 
                 layer_mid_attns_depth, attn_heads,
                 attend_at_middle):
        super().__init__()
        
        attn_kwargs = dict(heads=attn_heads)
        
        self.mid_block1 = ResnetBlock(mid_dim, 
                                      mid_dim, 
                                      cond_dim=cond_dim, 
                                      time_cond_dim = time_cond_dim, 
                                      groups = mid_resnet_group)
        self.mid_attn = TransformerBlock(mid_dim, depth=layer_mid_attns_depth,
                                         **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, 
                                      mid_dim, 
                                      cond_dim=cond_dim, 
                                      time_cond_dim=time_cond_dim, 
                                      groups = mid_resnet_group)

    def forward(self, x, t, c):
        x = self.mid_block1(x, time_emb=t, cond=c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, time_emb=t, cond=c)
        
        return x

class UpsamplingBlock(nn.Module):

    def __init__(self, dim_in, dim_out,
                 skip_connect_dim,
                 cond_dim, time_cond_dim,
                 attn_heads,
                 use_global_context_attn,
                 layer_num_resnet_blocks, 
                 groups, layer_attn, 
                 layer_attn_depth, layer_cross_attn, 
                 layer_use_linear_attn, 
                 layer_use_linear_cross_attn, 
                 ff_mult,
                 memory_efficient,
                 pixel_shuffle_upsample,
                 is_last_layer,
                 skip_connect_scale):
        super().__init__()
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample
        attn_kwargs = dict(heads=attn_heads)
        resnet_klass = partial(ResnetBlock, **attn_kwargs)
        self.skip_connect_scale = skip_connect_scale
        layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

        transformer_block_klass = Identity
        if layer_attn:
            transformer_block_klass = TransformerBlock
        elif layer_use_linear_attn:
            transformer_block_klass = LinearAttentionTransformerBlock
            
        self.us_block = nn.ModuleList([
            resnet_klass(dim_out+skip_connect_dim,
                         dim_out, 
                         cond_dim=layer_cond_dim, 
                         linear_attn=layer_use_linear_cross_attn, 
                         time_cond_dim=time_cond_dim, 
                         groups=groups),
            
            nn.ModuleList([ResnetBlock(dim_out+skip_connect_dim, 
                                       dim_out, 
                                       time_cond_dim=time_cond_dim, 
                                       groups=groups, 
                                       use_gca=use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
            
            transformer_block_klass(dim=dim_out, 
                                    depth=layer_attn_depth, 
                                    ff_mult=ff_mult, 
                                    context_dim=cond_dim,
                                    **attn_kwargs),
            
            upsample_klass(dim_out, dim_in) if not is_last_layer or memory_efficient else Identity()
        ])

    def forward(self, x, t, c, context, context_mask, down_hiddens, up_hiddens):
        
        init_block, resnet_blocks, attn_block, upsample = self.us_block
        x = torch.cat((x, down_hiddens.pop()*self.skip_connect_scale), dim=1)
        x = init_block(x, time_emb=t, cond=c)    # cross attention on given condition c

        for resnet_block in resnet_blocks:
            x = torch.cat((x, down_hiddens.pop()*self.skip_connect_scale), dim=1)
            x = resnet_block(x, time_emb=t) 

        x = attn_block(x, context, context_mask)       # self attention on concatenated c
        up_hiddens.append(x.contiguous())
        x = upsample(x)
        
        return x, down_hiddens, up_hiddens

class CondResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        use_gca = False,
    ):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x):

        h = self.block1(x)
        h = self.block2(h)
        h = h * self.gca(h)
        return h + self.res_conv(x).contiguous()

class CondEncoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out, 
                 layer_num_resnet_blocks, 
                 use_global_context_attn: bool,
                 groups: int,
                 memory_efficient: bool, 
                 is_last_layer: bool, 
                 cross_embed_downsample: bool,
                 cross_embed_downsample_kernel_sizes: tuple):
        
        super().__init__()

        # downsample klass define
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, 
                                       kernel_sizes=cross_embed_downsample_kernel_sizes)
        else:
            downsample_klass = Downsample

        if memory_efficient:
            # whether to pre-downsample, from memory efficient unet
            pre_downsample = downsample_klass(dim_in, dim_out)
            current_dim = dim_out
            post_downsample = None
            current_dim = dim_out
        else:
            # whether to do post-downsample, for non-memory efficient unet
            pre_downsample = None
            current_dim = dim_in

            if not is_last_layer:
                post_downsample = downsample_klass(current_dim, dim_out)  
            else: 
                post_downsample = Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))
        
        self.ds_block = nn.ModuleList([pre_downsample,
                                       CondResnetBlock(current_dim, current_dim, groups = groups),
                                       nn.ModuleList([CondResnetBlock(current_dim, current_dim,
                                                                      groups = groups, use_gca = use_global_context_attn
                                                                      ) for _ in range(layer_num_resnet_blocks)]),
                                       post_downsample])
        

    def forward(self, x):

        pre_downsample, init_block, resnet_blocks, post_downsample = self.ds_block
        
        if exists(pre_downsample):
            x = pre_downsample(x)

        x = init_block(x)

        for resnet_block in resnet_blocks:
            x = resnet_block(x)

        if exists(post_downsample):
            x = post_downsample(x)
        
        return x

class UNet2dBase(nn.Module):
    def __init__(
        self,
        dim,
        num_classes = 0,
        cond_drop_prob = 0.0,
        num_resnet_blocks = 1,
        cond_dim = None,
        num_time_tokens = 2,
        learned_sinu_pos_emb_dim = 16,
        dim_mults=[1, 2, 4, 8],
        channels = 3,
        channels_out = None,
        attn_heads = 8,
        ff_mult = 2.,
        layer_attns = True,
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        attend_at_middle = True,            
        # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading, before bringing in efficient attention)
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        text_embed_dim = 768, # Bert embedding dim
        class_embed_dim = None, 
        cond_on_text = False,
        max_text_len = 3,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,   # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        resize_mode = 'nearest',
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
        use_condition_block = False, 
        channel_infuse_mode = None,
    ):
        super().__init__()
        self.use_condition_block = use_condition_block

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'
        assert dim > 100 #128

        # determine dimensions
        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = channels
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # initial convolution
        if init_cross_embed:
            self.init_conv = CrossEmbedLayer(init_channels, dim_out=init_dim, 
                                             kernel_sizes=init_cross_embed_kernel_sizes, 
                                             stride=1)
        else:
            self.init_conv = nn.Conv2d(init_channels, init_dim, 
                                       init_conv_kernel_size, 
                                       padding=init_conv_kernel_size//2)
        self.init_conv_cond = copy.deepcopy(self.init_conv) if use_condition_block else None


        # time conditioning
        cond_dim = default(cond_dim, dim)
        time_cond_dim = cond_dim * 4
        self.cond_drop_prob = cond_drop_prob 

        # embedding time for log(snr) noise from continuous version
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # project to time tokens as well as time hiddens
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        # class embeds
        self.num_classes = num_classes
        if self.num_classes != 0:

            classes_dim = dim * 4

            assert num_classes is not None or class_embed_dim is not None
            self.label_conditioner = LabelEmbedder(num_classes, 
                                                   class_embed_dim, 
                                                   dim, 
                                                   classes_dim)

        # text encoding conditioning (optional)
        self.cond_on_text = cond_on_text
        if cond_on_text:
            self.text_conditioner = TextEmbedder(cond_dim, text_embed_dim, max_text_len)

        # attention related params
        attn_kwargs = dict(heads = attn_heads)
        num_layers = len(in_out)

        # resnet block klass params
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)
        
        layer_attns = cast_tuple(list(layer_attns))
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(list(layer_cross_attns))

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers==num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # initial resnet block (for memory efficient unet)
        self.init_resnet_block = None
        if memory_efficient:
            self.init_resnet_block = resnet_klass(init_dim, init_dim, 
                                                  time_cond_dim=time_cond_dim, 
                                                  groups=resnet_groups[0], 
                                                  use_gca=use_global_context_attn)
        self.init_resnet_block_cond = copy.deepcopy(self.init_resnet_block) if use_condition_block else None

        # scale for resnet skip connections
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers
        self.downs = nn.ModuleList([])
        self.downs_cond = nn.ModuleList([]) if use_condition_block else None
        num_resolutions = len(in_out)
        layer_params = [num_resnet_blocks, 
                        resnet_groups, 
                        layer_attns, 
                        layer_attns_depth, 
                        layer_cross_attns, 
                        use_linear_attn, 
                        use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers
        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            
            is_last = ind >= (num_resolutions - 1)
            ds_block = DownsamplingBlock(dim_in, dim_out,
                                         cond_dim,
                                         time_cond_dim,
                                         attn_heads,
                                         use_global_context_attn,
                                         layer_num_resnet_blocks, 
                                         groups, 
                                         layer_attn, 
                                         layer_attn_depth, 
                                         layer_cross_attn, 
                                         layer_use_linear_attn, 
                                         layer_use_linear_cross_attn, 
                                         ff_mult,
                                         memory_efficient, 
                                         is_last, 
                                         cross_embed_downsample,
                                         cross_embed_downsample_kernel_sizes, 
                                         channel_infuse_mode)
            current_dim = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(current_dim)
            self.downs.append(ds_block)
            if use_condition_block:
                ds_block_cond = CondEncoderBlock(dim_in, dim_out, 
                                                 layer_num_resnet_blocks, 
                                                 use_global_context_attn,
                                                 groups, memory_efficient, is_last, 
                                                 cross_embed_downsample,
                                                 cross_embed_downsample_kernel_sizes)    
                self.downs_cond.append(ds_block_cond)

        self.mid_block = MiddleBlock(dims[-1],
                                     cond_dim, 
                                     time_cond_dim, 
                                     resnet_groups[-1], 
                                     layer_mid_attns_depth, 
                                     attn_heads,
                                     attend_at_middle)

        # upsample klass
        self.ups = nn.ModuleList([])

        # upsampling layers
        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):

            skip_connect_dim = skip_connect_dims.pop()
            is_last = ind == (len(in_out) - 1)
            self.ups.append(UpsamplingBlock(dim_in, dim_out,
                                            skip_connect_dim,
                                            cond_dim, time_cond_dim,
                                            attn_heads,
                                            use_global_context_attn,
                                            layer_num_resnet_blocks, 
                                            groups, layer_attn, 
                                            layer_attn_depth, layer_cross_attn, 
                                            layer_use_linear_attn, 
                                            layer_use_linear_cross_attn, 
                                            ff_mult,
                                            memory_efficient,
                                            pixel_shuffle_upsample,
                                            is_last,
                                            self.skip_connect_scale))

        # whether to combine feature maps from all upsample blocks before final resnet block out
        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )

        # whether to do a final residual from initial conv to the final resnet block out
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out
        
        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim=time_cond_dim, groups=resnet_groups[0], use_gca=True) if final_resnet_block else None
        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        self.final_conv = nn.Conv2d(final_conv_dim_in, 
                                    self.channels_out, 
                                    final_conv_kernel_size, 
                                    padding=final_conv_kernel_size//2)

        nn.init.zeros_(self.final_conv.weight)
        if exists(self.final_conv.bias):
            nn.init.zeros_(self.final_conv.bias)

        # resize mode
        self.resize_mode = resize_mode

    def forward(self, 
                x, time,
                classes=None,
                text_embeds = None,
                text_mask = None,
                cond_drop_prob=None,
                inj_channels: Optional[Tensor] = None):
        
        batch_size, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        
        # initial convolution
        x = self.init_conv(x)
        if inj_channels is not None and self.use_condition_block:
            inj_channels = self.init_conv_cond(inj_channels)
            
        # init conv residual
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning
        time_hiddens = self.to_time_hiddens(time)

        # derive time tokens, where tokens are used for cross attention conditions
        t = self.to_time_cond(time_hiddens)

        ## cfg condition for class labels, with condition dropout for classifier free guidance
        if self.num_classes != 0:
            assert classes is not None # only applies to conditional cases
            classes_emb = self.label_conditioner(classes, cond_drop_prob)
            t = t + classes_emb

        ## cfg condition for text embeds, with condition dropout for classifier free guidance
        if exists(text_embeds):
            context, context_mask = self.text_conditioner(text_embeds, text_mask, cond_drop_prob)  
            
        else:
            context = None
            context_mask = None

        # initial resnet block (for memory efficient unet)
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)
            if inj_channels is not None and self.use_condition_block:
                inj_channels = self.init_resnet_block_cond(inj_channels)

        # go through the layers of the unet, down and up
        hiddens = []
        if self.use_condition_block and inj_channels is not None:
            inj_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
                
            for cond_block, block in zip(self.downs_cond, self.downs):

                inj_channels[inj_keep_mask == False] = 0

                x, hiddens = block(x, t=t, c=None, 
                                   context=context, 
                                   context_mask=context_mask, 
                                   hiddens=hiddens, 
                                   inj_channels=inj_channels)
                inj_channels = cond_block(inj_channels)

        else:
            for block in self.downs:
                x, hiddens = block(x, t, c=None, 
                                   context=context, 
                                   context_mask=context_mask, 
                                   hiddens=hiddens)

        x = self.mid_block(x, t, c=None)
        
        up_hiddens = []
        for block in self.ups:
            x, hiddens, up_hiddens = block(x, t, c=None, 
                                           context=context, 
                                           context_mask=context_mask, 
                                           down_hiddens=hiddens, 
                                           up_hiddens=up_hiddens)

        assert len(hiddens) == 0

        # whether to combine all feature maps from upsample blocks
        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed
        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim=1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        return self.final_conv(x)
