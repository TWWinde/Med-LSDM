"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import os.path

import numpy as np
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_3d.model.lpips import LPIPS
from vq_gan_3d.model.codebook import Codebook
from evaluation.metrics_vq_gan import metrics

def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class BCELossWithQuant(nn.Module):
    def __init__(self, image_channels=159, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.register_buffer(
            "weight",
            torch.ones(image_channels).index_fill(0, torch.arange(153, 158), 20),
        )

    def forward(self, qloss, target, prediction):
        bce_loss = F.binary_cross_entropy_with_logits(
            prediction.permute(0, 2, 3, 1),
            target.permute(0, 2, 3, 1),
            pos_weight=self.weight,
        )
        loss = bce_loss + self.codebook_weight * qloss
        return loss


class VQVAE(pl.LightningModule):
    def __init__(self, cfg, val_dataloader=None):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.padding_type,
                               cfg.model.num_groups,
                               )
        self.decoder = Decoder(
            cfg.model.n_hiddens, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, cfg.model.embedding_dim, 1, padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv3d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)

        self.gan_feat_weight = cfg.model.gan_feat_weight


        self.l1_weight = cfg.model.l1_weight
        self.save_hyperparameters()
        self.num_classes = cfg.dataset.image_channels
        self.val_dataloader = val_dataloader
        #self.path = os.path.join(self.cfg.model.default_root_dir, self.cfg.model.name, self.cfg.model.default_root_dir_postfix, 'metrics')
        #os.makedirs(self.path, exist_ok=True)
        #self.metrics_computer = metrics(self.path, self.val_dataloader, self.num_classes)

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

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, log_image=False, evaluation=False):

        x = x.cuda()

        x = self.preprocess_input(x)

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))  # torch.Size([B, 37, 32, 256, 256]) for seg

        #if self.global_step % 500 == 0:
            #self.metrics_computer.update_metrics(x, x_recon, self.global_step)

        bce_loss = F.binary_cross_entropy_with_logits(
                x_recon.permute(0, 2, 3, 1),
                x.permute(0, 2, 3, 1),
                pos_weight=torch.ones(self.num_classes).index_fill(0, 0, 0.05),
            )

        if log_image:
            return x_recon, z, vq_output

        self.log("train/binary_crossentropy_loss", bce_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
        self.log("train/commitment_loss", vq_output['commitment_loss'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/perplexity', vq_output['perplexity'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return x_recon, bce_loss, vq_output,

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['image']

        _, bce_loss, vq_output = self.forward(x, optimizer_idx)
        commitment_loss = vq_output['commitment_loss']
        loss = bce_loss + commitment_loss

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']  # TODO: batch['stft']
        _, bce_loss, vq_output = self.forward(x)
        self.log("train/binary_crossentropy_loss", bce_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/commitment_loss", vq_output['commitment_loss'],
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/perplexity', vq_output['perplexity'],
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        return [opt_ae], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['image']
        x = x.to(self.device)
        frames, frames_rec, _, _ = self.forward(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch['image']
        _, _, x, x_rec = self.forward(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _
