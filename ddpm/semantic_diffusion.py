"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import copy
import os.path
import random

import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from ddpm.text import tokenize, bert_embed, BERT_MODEL_DIM
from torch.utils.data import Dataset, DataLoader
from vq_gan_3d.model.vqgan import VQGAN
from vq_gan_3d.model.vqgan_spade import VQGAN_SPADE
from vq_gan_3d.model.vqvae import VQVAE
from evaluation.metrics import Metrics
import matplotlib.pyplot as plt


# helpers functions


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


# relative positional bias


class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance /
                                                            max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class SPADEGroupNorm3D(nn.Module):
    def __init__(self, dim_out, spade_input_nc=64, eps=1e-5, groups=8, dim_hidden=128):  # !!! dim_hidden ????
        super().__init__()

        self.norm = nn.GroupNorm(groups, dim_out, affine=False)
        self.eps = eps

        self.mlp_shared = nn.Sequential(
            nn.Conv3d(spade_input_nc, dim_hidden, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv3d(dim_hidden, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.mlp_beta = nn.Conv3d(dim_hidden, dim_out, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, x, segmap):
        x = self.norm(x)
        # seg torch.Size([10, 37, 32, 256, 256])
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')  # !!! is 2 right?
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return x * (1 + gamma) + beta


class SDDResBlock(nn.Module):
    """
    Semantic Diffusion Decoder Resnet Block
    scale_shift for time embedding

    """

    def __init__(self, dim, dim_out, time_emb_dim=None, spade_input_nc=37, groups=8):
        super().__init__()
        self.spade_input_nc = spade_input_nc
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.conv1 = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(dim_out, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.spade_norm1 = SPADEGroupNorm3D(dim, spade_input_nc=self.spade_input_nc, eps=1e-5, groups=groups)
        self.spade_norm2 = SPADEGroupNorm3D(dim_out, spade_input_nc=self.spade_input_nc, eps=1e-5, groups=groups)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, seg, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.spade_norm1(x, seg)
        h = self.act(h)
        h = self.conv1(h)
        h = self.spade_norm2(h, seg)

        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.act(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class SDEResBlock(nn.Module):
    """
     Semantic Diffusion Encoder Resnet Block
     scale_shift for time embedding

    """

    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.conv1 = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(dim_out, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm1 = nn.GroupNorm(groups, dim)
        self.norm2 = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.act(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    """
     block = 3D Conv + Group norm + (scale_shift) + SiLU
     scale_shift for time embedding

    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    """
     block = 3D Conv + Group norm + (scale_shift) + SiLU
     ResnetBlock = 2* block + residual connection
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class SegConv3D(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(SegConv3D, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=64, kernel_size=(1, 3, 3), stride=1,
                               padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce (32, 256, 256) to (16, 128, 128)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Reduce (16, 128, 128) to (8, 64, 64)

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=self.output_channels, kernel_size=(1, 3, 3), stride=1,
                               padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)

        return x


class Unet3D_SPADE(nn.Module):
    def __init__(
            self,
            dim,  # 64
            cond_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,  # 8
            attn_heads=8,
            attn_dim_head=32,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            label_nc=37,
            spade_input_nc=64,
            block_type='resnet',
            resnet_groups=8,
            segconv=False,
            vqvae=None,
            add_seg_to_noise=False
    ):
        super().__init__()
        self.channels = channels

        self.label_nc = label_nc
        if segconv:
            self.spade_input_nc = spade_input_nc
        elif vqvae is not None:
            self.spade_input_nc = 8
        else:
            self.spade_input_nc = self.label_nc

        self.add_seg_to_noise = add_seg_to_noise

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim):
            return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
                dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        if self.add_seg_to_noise:
            self.init_conv = nn.Conv3d(self.channels+self.label_nc, init_dim, (1, init_kernel_size,init_kernel_size),
                                       padding=(0, init_padding, init_padding))   ####### self.spade_input_nc
        else:
            self.init_conv = nn.Conv3d(self.channels, init_dim, (1, init_kernel_size,
                                                                 init_kernel_size),
                                       padding=(0, init_padding, init_padding))


        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        # [64, 64, 128, 256]  ,  [64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]  # [64, 64, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(64, 64), (64, 128), (128, 256), (256, 512)]

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type  replaced by SDEResBlock and SDDResBlock

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):  # [(64, 64), (64, 128), (128, 256), (256, 512)]
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                SDEResBlock(dim_in, dim_out, time_emb_dim=cond_dim, groups=resnet_groups),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = SDDResBlock(mid_dim, mid_dim, spade_input_nc=self.spade_input_nc)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = SDDResBlock(mid_dim, mid_dim, spade_input_nc=self.spade_input_nc)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                SDDResBlock(dim_out * 2, dim_in, time_emb_dim=cond_dim, groups=resnet_groups,
                            spade_input_nc=self.spade_input_nc),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

        # add 3d conv for input segmap

        self.segconv = segconv
        if self.segconv:
            self.segconv3d = SegConv3D(self.label_nc, self.spade_input_nc)
        else:
            self.segconv3d = None

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.5,
            cond,
            **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., cond=cond, **kwargs)
        # if cond_scale == 1:  # or not self.has_cond:
        # return logits
        all_zero = torch.zeros_like(cond)
        null_logits = self.forward(*args, null_cond_prob=1., cond=all_zero, **kwargs)
        return logits + (logits - null_logits) * cond_scale

    def segconv_downsample(self, seg_input):

        return self.segconv3d(seg_input)

    def forward(
            self,
            x,
            time,
            cond=None,
            null_cond_prob=0.,
            focus_present_mask=None,
            prob_focus_present=0.
    ):
        # assert not (self.has_cond and not exists(cond) ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # add 3d conv for input segmap
        # print(cond.shape) torch.Size([1, 37, 32, 256, 256])
        if self.segconv:
            seg = self.segconv3d(cond)
        else:
            seg = cond

        h = []

        for block, spatial_attn, temporal_attn, downsample in self.downs:
            x = block(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, seg, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, seg, t)

        for block, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, seg, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# model
# gaussian diffusion trainer class


def extract(a, t, x_shape):
    """
    get elements from list
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # (b, 1, 1, ..., 1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class SemanticGaussianDiffusion(nn.Module):
    def __init__(
            self,
            cfg,
            denoise_fn,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9,
            vqgan_ckpt=None,
            vqgan_spade_ckpt=None,
            vqvae_ckpt=None,
            cond_scale=1.5,
            add_seg_to_noise=False
    ):
        super().__init__()
        self.cfg = cfg
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn  # Unet3D noise_predictor
        self.cond_scale = cond_scale

        if vqgan_ckpt and not vqgan_spade_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
            self.vqgan_spade = None
            print("########using vqgan to mapping########")
        elif vqgan_spade_ckpt:
            self.vqgan_spade = VQGAN_SPADE.load_from_checkpoint(vqgan_spade_ckpt).cuda()
            self.vqgan_spade.eval()
            self.vqgan = None
            print("########using vqgan_spade to mapping########")
        elif vqvae_ckpt:
            self.vqvae = VQVAE.load_from_checkpoint(vqgan_spade_ckpt).cuda()
            self.vqvae.eval()
            self.vqgan = None

            print("########using vqvae to mapping########")

        else:
            self.vqgan = None
            self.vqgan_spade = None
            self.vqvae = None
            print("######## vqgan vqvae vqgan_spade models are not implemented########")

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.add_seg_to_noise = add_seg_to_noise
        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(
                name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)  # alphas_t_bar
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # alphas_t-1_bar

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # sqrt(alphas_t_bar)
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))  # sqrt(1-alphas_t_bar)
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))  # log(1-alphas_t_bar)
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))  # sqrt(1/alphas_t_bar)
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))  # sqrt((1/alphas_t_bar)-1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_sample(self, x_start, t, noise=None):  # x_{t} get noisy images
        """
            Diffuse the data for a given number of diffusion steps.

            In other words, sample from q(x_t | x_0).

            :param x_start: the initial data batch.
            :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
            :param noise: if specified, the split-out normal noise.
            :return: A noisy version of x_start.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod,
                        t, x_start.shape) * noise
        )

        return x_t

    def q_mean_variance(self, x_start, t):  # q(x_{t} | x_0)
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):  # x_{0} = f(x_{t}, noise, t)

        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

                 q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.5, seg=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
                   x_start prediction before it is used to sample. Applies before
                   clip_denoised.

        :return: a dict with the following keys:
                    - 'mean': the model mean output.
                    - 'variance': the model variance output.
                    - 'log_variance': the log of 'variance'.
         """

        if self.add_seg_to_noise:

            cond = F.interpolate(cond, size=(8, 64, 64), mode='trilinear', align_corners=False)
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn.forward_with_cond_scale(torch.cat([x, cond], 1), t, cond=cond, cond_scale=cond_scale))
            pass
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale))

        # elf.denoise_fn == Unet3D   x_recon == x_{0}

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)  # q(x_{t-1} |x_{t}, x_0)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()  # add noise
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True, seg=None):
        """
        Sample x_{t-1} from the model at the given timestep. from noise

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.

        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = \
            self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale, seg=seg)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise  # x_{t-1}

    number = 1

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.5, get_middle_process=False):
        """
            Generate samples from the model. From noise to x_{t}

        """

        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        random_number = random.randint(0, 100)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)

            if get_middle_process:
                if i % 25 == 0:
                    if isinstance(self.vqgan, VQGAN):
                        # denormalize TODO: Remove eventually
                        img_save = (((img + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                           self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

                        img_save = self.vqgan.decode(img_save, quantize=True)
                    elif isinstance(self.vqgan_spade, VQGAN_SPADE):
                        # denormalize TODO: Remove eventually
                        img_save = (((img + 1.0) / 2.0) * (self.vqgan_spade.codebook.embeddings.max() -
                                                           self.vqgan_spade.codebook.embeddings.min())) + self.vqgan_spade.codebook.embeddings.min()

                        img_save = self.vqgan_spade.decode(img_save, cond, quantize=True)
                    elif isinstance(self.vqvae, VQVAE):
                        # denormalize TODO: Remove eventually
                        img_save = (((img + 1.0) / 2.0) * (self.vqvae.codebook.embeddings.max() -
                                                           self.vqvae.codebook.embeddings.min())) + self.vqvae.codebook.embeddings.min()

                        img_save = self.vqvae.decode(img_save, quantize=True)
                    else:
                        unnormalize_img(img)

                    #img_save = F.pad(img_save, (2, 2, 2, 2))

                    slice_index = 16
                    image_np = img_save.cpu().numpy()

                    results_folder = os.path.join("/data/private/autoPET/medicaldiffusion_results/",
                                                  self.cfg.model.name,
                                                  self.cfg.dataset.name, "diffusion_middle_process_slices")
                    os.makedirs(results_folder, exist_ok=True)
                    # Sample slice
                    plt.imshow(image_np[0, 0, slice_index, :, :], cmap='gray') # Choose the colormap as needed
                    plt.axis('off')  # Optional: Turn off axes
                    plt.savefig(os.path.join(results_folder, f'{random_number}_{500-i}_sample_slice_{slice_index}.png'), bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    #os.makedirs(results_folder, exist_ok=True)

                    #sample_path = os.path.join(results_folder, f'{random_number}_{480 - i}_sample.gif')
                    #video_tensor_to_gif(sample_gif, sample_path)

        print('#################### sample finished ####################')

        return img

    @torch.inference_mode()
    def sample(self, cond=None, batch_size=16, get_middle_process=False):
        """
           from latent space to image space

        """
        # device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        # cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        _sample = self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=self.cond_scale,
            get_middle_process=get_middle_process)

        assert not isinstance(self.vqgan, VQGAN) or not isinstance(self.vqgan_spade, VQGAN_SPADE)
        if isinstance(self.vqgan, VQGAN):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                  self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

            _sample = self.vqgan.decode(_sample, quantize=True)
        elif isinstance(self.vqgan_spade, VQGAN_SPADE):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan_spade.codebook.embeddings.max() -
                                                  self.vqgan_spade.codebook.embeddings.min())) + self.vqgan_spade.codebook.embeddings.min()

            _sample = self.vqgan_spade.decode(_sample, cond, quantize=True)
        elif isinstance(self.vqvae, VQVAE):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (self.vqvae.codebook.embeddings.max() -
                                                  self.vqvae.codebook.embeddings.min())) + self.vqvae.codebook.embeddings.min()

            _sample = self.vqvae.decode(_sample, quantize=True)

        else:
            unnormalize_img(_sample)

        return _sample

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        # b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # x_{t}
        """
         if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        """
        #torch.Size([1, 8, 8, 64, 64])
        #torch.Size([1, 37, 32, 256, 256])
        #print(cond.shape)
        if self.add_seg_to_noise:
            #cond = self.denoise_fn.segconv_downsample(cond)  # torch.Size([1, 64, 11, 64, 64])
            #print(x_noisy.shape)
            #print(cond.shape)
            cond = F.interpolate(cond, size=(8, 64, 64), mode='trilinear', align_corners=False)
            noise_predicted = self.denoise_fn(torch.cat([x_noisy, cond], 1), t, cond=cond, **kwargs)

        else:
            noise_predicted = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)  # predicted noise


        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, noise_predicted)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_predicted)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, cond, **kwargs):

        assert not isinstance(self.vqgan, VQGAN) or not isinstance(self.vqgan_spade, VQGAN_SPADE)
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                x = self.vqgan.encode(x, quantize=False, include_embeddings=True)
                # normalize to -1 and 1
                x = ((x - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() -
                      self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0

        elif isinstance(self.vqgan_spade, VQGAN_SPADE):
            with torch.no_grad():
                x = self.vqgan_spade.encode(
                    x, quantize=False, include_embeddings=True)
                # normalize to -1 and 1
                x = ((x - self.vqgan_spade.codebook.embeddings.min()) /
                     (self.vqgan_spade.codebook.embeddings.max() -
                      self.vqgan_spade.codebook.embeddings.min())) * 2.0 - 1.0
        elif isinstance(self.vqvae, VQVAE):
            with torch.no_grad():
                x = self.vqvae.encode(
                    x, quantize=False, include_embeddings=True)
                # normalize to -1 and 1
                x = ((x - self.vqvae.codebook.embeddings.min()) /
                     (self.vqvae.codebook.embeddings.max() -
                      self.vqvae.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            print("Hi")
            x = normalize_img(x)  # (-1,1)

        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, cond, *args, **kwargs)


# trainer class


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
            self,
            folder,
            image_size,
            channels=3,
            num_frames=16,
            horizontal_flip=False,
            force_num_frames=True,
            exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)


# trainer class


class Semantic_Trainer(object):
    def __init__(
            self,
            diffusion_model,
            cfg,
            folder=None,
            train_dataset=None,
            val_dataset=None,
            *,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            num_sample_rows=1,
            max_grad_norm=None,
            num_workers=20,
            vqvae_ckpt=None,
            vqgan_spade_ckpt
    ):
        super().__init__()
        self.cfg = cfg
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if vqvae_ckpt:
            self.vqvae = VQVAE.load_from_checkpoint(vqvae_ckpt).cuda()
            self.vqvae.eval()
            print('vqvae is implemented')
        else:
            self.vqvae = None
        self.vqgan_spade_ckpt = vqgan_spade_ckpt
        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames
        self.num_classes = self.cfg.dataset.label_nc

        self.cfg = cfg
        if train_dataset:
            self.ds = train_dataset
        else:
            assert folder is not None, 'Provide a folder path to the dataset'
            self.ds = Dataset(folder, image_size,
                              channels=channels, num_frames=num_frames)
        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)
        val_dl = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=num_workers)

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(
            self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.metrics_folder = os.path.join(self.results_folder)
        os.makedirs(self.metrics_folder, exist_ok=True)
        self.metrics_computer = Metrics(self.metrics_folder, val_dl, num_classes=self.num_classes)
        self.reset_parameters()

    def preprocess_input(self, data):

        # move to GPU and change data types
        data = data.long()

        # create one-hot label map
        label_map = data
        bs, _, t, h, w = label_map.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        return input_semantics

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        path_cp = os.path.join(self.results_folder, 'checkpoints')
        os.makedirs(path_cp, exist_ok=True)
        path = os.path.join(path_cp, f'sample-{milestone}.pt')
        torch.save(data, path)

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_paths = os.listdir(os.path.join(self.results_folder, 'checkpoints'))
            all_milestones = [int((p.split('.')[0]).split("-")[-1]) for p in all_paths if p.endswith('.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            print('found checkpoint', os.path.join(self.results_folder, 'checkpoints', f"sample-{milestone}.pt"))
            data = torch.load(os.path.join(self.results_folder, 'checkpoints', f"sample-{milestone}.pt"))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])
        print("checkpoint is successful loaded")

    def save_image(self, image_tensor, path, cols=3):
        B, C, H, W = image_tensor.shape
        plt.figure(figsize=(50, 50))
        for i in range(B):
            plt.subplot(B // cols + 1, cols, i + 1)
            img = image_tensor[i].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
            plt.imshow(img, cmap='gray' if C == 1 else None)
            plt.axis('off')
        plt.savefig(path)
        plt.close()

    def train(self, prob_focus_present=0., focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data_ = next(self.dl)
                input_image, label = data_['image'].cuda(), data_['label'].cuda()
                seg = self.preprocess_input(label)

                if isinstance(self.vqvae, VQVAE):
                    with torch.no_grad():
                        seg = self.vqvae.encode(seg, quantize=False, include_embeddings=True)
                        # normalize to -1 and 1
                        seg = ((seg - self.vqvae.codebook.embeddings.min()) /
                               (self.vqvae.codebook.embeddings.max() -
                                self.vqvae.codebook.embeddings.min())) * 2.0 - 1.0
                        assert seg.size()[-1] == 64  # torch.Size([1, 8, 8, 64, 64])

                with autocast(enabled=self.amp):
                    if self.vqgan_spade_ckpt:
                        loss = self.model(
                            input_image,
                            cond=seg,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask
                        )
                    else:
                        loss = self.model(
                            input_image,
                            cond=seg,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask
                        )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                if self.step % 50 == 0:
                    print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema_model.eval()
                milestone = self.step // self.save_and_sample_every
                # update metrics
                self.metrics_computer.update_metrics(self.ema_model, milestone, encoder=self.vqvae)

                with torch.no_grad():
                    num_samples = self.num_sample_rows ** 2
                    batches = num_to_groups(num_samples, self.batch_size)

                    all_videos_list = list(
                        map(lambda n: self.ema_model.sample(cond=seg, batch_size=n), batches))
                    all_videos_list = torch.cat(all_videos_list, dim=0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
                all_label_list = F.pad(label, (2, 2, 2, 2))
                all_image_list = F.pad(input_image, (2, 2, 2, 2))
                if self.step != 0 and self.step % (self.save_and_sample_every * 5) == 0:
                    sample_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                    label_gif = rearrange(all_label_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                    image_gif = rearrange(all_image_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                    path_video = os.path.join(self.results_folder, 'video_results')
                    os.makedirs(path_video, exist_ok=True)

                    sample_path = os.path.join(path_video, f'{milestone}_sample.gif')
                    image_path = os.path.join(path_video, f'{milestone}_image.gif')
                    label_path = os.path.join(path_video, f'{milestone}_label.gif')
                    video_tensor_to_gif(sample_gif, sample_path)
                    video_tensor_to_gif(image_gif, image_path)
                    video_tensor_to_gif(label_gif, label_path)
                    log = {**log, 'sample': sample_path}

                # Selects one random 2D image from each 3D Image
                B, C, D, H, W = all_videos_list.shape
                frame_idx = torch.randint(0, D, [B]).cuda()
                frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                sample_frames = torch.gather(all_videos_list, 2, frame_idx_selected).squeeze(2)
                label_frames = torch.gather(all_label_list, 2, frame_idx_selected).squeeze(2)
                image_frames = torch.gather(all_image_list, 2, frame_idx_selected).squeeze(2)
                path_image_root = os.path.join(self.results_folder, 'images_results')
                os.makedirs(path_image_root, exist_ok=True)
                path_sampled = os.path.join(path_image_root, f'{milestone}-sample.jpg')
                path_label = os.path.join(path_image_root, f'{milestone}-label.jpg')
                path_image = os.path.join(path_image_root, f'{milestone}-image.jpg')

                def save_image(image_tensor, path, cols=2):
                    B, C, H, W = image_tensor.shape
                    plt.figure(figsize=(50, 50))
                    for i in range(B):
                        plt.subplot(B // cols + 1, cols, i + 1)
                        img = image_tensor[i].cpu().numpy().transpose(1, 2, 0)
                        img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
                        plt.imshow(img, cmap='gray' if C == 1 else None)
                        plt.axis('off')
                    plt.savefig(path)
                    plt.close()

                save_image(sample_frames, path_sampled)
                save_image(label_frames, path_label)
                save_image(image_frames, path_image)

                if self.step != 0 and self.step % (self.save_and_sample_every * 50) == 0:
                    self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')
