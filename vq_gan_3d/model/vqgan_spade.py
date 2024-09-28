import math
import argparse
import os

import numpy as np
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

#from evaluation.metrics_vq_gan import metrics
from vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_3d.model.lpips import LPIPS
from vq_gan_3d.model.codebook import Codebook


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


class VQGAN_SPADE(pl.LightningModule):
    def __init__(self, cfg,  val_dataloader=None, label=False):
        super().__init__()
        self.cfg = cfg
        self.label = label
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes
        self.label_nc = cfg.dataset.label_nc

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.padding_type,
                               cfg.model.num_groups,
                               )
        self.decoder = Decoder(
            cfg.model.n_hiddens, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups, self.label_nc)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, cfg.model.embedding_dim, 1, padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv3d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm3d)

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = cfg.model.image_gan_weight
        self.video_gan_weight = cfg.model.video_gan_weight
        if self.label:
            self.perceptual_weight = 0
        else:
            self.perceptual_weight = cfg.model.perceptual_weight

        self.l1_weight = cfg.model.l1_weight
        self.save_hyperparameters()
        self.num_classes = cfg.dataset.label_nc
        self.path = os.path.join(self.cfg.model.default_root_dir, self.cfg.model.name, self.cfg.model.default_root_dir_postfix, 'metrics')
        os.makedirs(self.path, exist_ok=True)
        #self.metrics_computer = metrics(self.path, self.val_dataloader, self.num_classes)

    """
    def preprocess_input(self, data):

        label = data['label'].long()
        img = data['image'].float()
        # create one-hot label map
        bs, _, t, h, w = label.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label, 1.0)

        return img, input_semantics

    """

    def preprocess_input(self, data):

        data = data.long()
        # create one-hot label map
        label_map = data
        bs, _, t, h, w = label_map.size()
        nc = self.num_classes
        assert label_map.max() < self.num_classes, f"Label map contains indices {label_map.max()} greater than number of classes {self.num_classes}"
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

    def decode(self, latent, seg, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h, seg)

    def forward(self, x, seg, optimizer_idx=None, log_image=False, evaluation=False):

        #B, C, T, H, W = x.shape  # ([2, 1, 32, 256, 256])
        if self.label:
            x = self.preprocess_input(x)

        B, C, T, H, W = x.shape
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']), seg)  # torch.Size([B, 37, 32, 256, 256]) for seg

        #if self.global_step % 500 == 0:
            #self.metrics_computer.update_metrics(x, x_recon, self.global_step)

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        if self.label:
            recon_labels = torch.argmax(x_recon, dim=1)  # Convert to class indices
            target_labels = torch.argmax(x, dim=1)  # Convert to class indices
            criterion = torch.nn.CrossEntropyLoss()
            crossentropy_loss = criterion(recon_labels, target_labels)
        else:
            crossentropy_loss = 0

            # Selects one random 2D image from each 3D Image

        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if evaluation:
            return x_recon, z, vq_output

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"

            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(
                    frames, frames_recon).mean() * self.perceptual_weight

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(
                frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(
                x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(
                    frames)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                        )) * (self.image_gan_weight > 0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(
                    x)
                for i in range(len(pred_video_fake)-1):
                    video_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_video_fake[i], pred_video_real[i].detach(
                        )) * (self.video_gan_weight > 0)
            gan_feat_loss = disc_factor * self.gan_feat_weight * \
                (image_gan_feat_loss + video_gan_feat_loss)

            self.log("train/g_image_loss", g_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/g_video_loss", g_video_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/video_gan_feat_loss", video_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/crossentropy_loss", crossentropy_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, crossentropy_loss, gan_feat_loss

        if optimizer_idx == 1:
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(
                frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start)
            discloss = disc_factor * \
                (self.image_gan_weight*d_image_loss +
                 self.video_gan_weight*d_video_loss)

            self.log("train/logits_image_real", logits_image_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake", logits_image_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_real", logits_video_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_video_fake", logits_video_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_video_loss", d_video_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)

            return discloss

        if self.label:
            perceptual_loss = 0
        else:
            perceptual_loss = self.perceptual_model(
                frames, frames_recon) * self.perceptual_weight

        return recon_loss, crossentropy_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        #x, seg = self.preprocess_input(batch)
        x, label = batch['image'], batch['label']
        seg = self.preprocess_input(label)

        if optimizer_idx == 0:
            recon_loss, _, vq_output, aeloss, perceptual_loss, crossentropy_loss, gan_feat_loss = self.forward(
                x, seg, optimizer_idx)
            commitment_loss = vq_output['commitment_loss']
            loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss + crossentropy_loss
        if optimizer_idx == 1:
            discloss = self.forward(x, seg, optimizer_idx)
            loss = discloss

        return loss

    def validation_step(self, batch, batch_idx):

        #x, seg = self.preprocess_input(batch)
        x, label = batch['image'], batch['label']
        seg = self.preprocess_input(label)

        recon_loss, crossentropy_loss, _, vq_output, perceptual_loss = self.forward(x, seg)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/crossentropy_loss', crossentropy_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        #x = batch['image']
        #x, seg = self.preprocess_input(batch)
        #x = x.to(self.device)
        #seg = seg.to(self.device)
        x, label = batch['image'].cuda(), batch['label'].cuda()
        seg = self.preprocess_input(label)
        frames, frames_rec, _, _ = self.forward(x, seg, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        #x = batch['image']
        #x, seg = self.preprocess_input(batch)
        #x = x.to(self.device)
        x, label = batch['image'].cuda(), batch['label'].cuda()
        seg = self.preprocess_input(label)
        _, _, x, x_rec = self.forward(x, seg, log_image=True)
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


class SPADEGroupNorm3D(nn.Module):
    def __init__(self, dim_out, label_nc=37, eps=1e-5, groups=8, dim_hidden=128):   # !!! dim_hidden ????
        super().__init__()

        self.norm = nn.GroupNorm(groups, dim_out, affine=False)
        self.eps = eps

        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, dim_hidden, (1, 3, 3), padding=(0, 1, 1)),
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


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32, label_nc=37):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.label_nc = label_nc
        in_channels = n_hiddens*2**max_us
        self.final_block_spade = SPADEGroupNorm3D(in_channels, label_nc=self.label_nc)  # SiLU())

        self.conv_blocks_spade = nn.ModuleList()
        for i in range(max_us):
            block_spade = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block_spade.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
            block_spade.res1 = Spade_ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups, label_nc=self.label_nc)
            block_spade.res2 = Spade_ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups, label_nc=self.label_nc)
            self.conv_blocks_spade.append(block_spade)
            n_times_upsample -= 1

        self.conv_last_spade = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)

    def forward(self, x, seg):
        h = self.final_block_spade(x, seg)
        h = silu(h)
        for i, block_spade in enumerate(self.conv_blocks_spade):
            h = block_spade.up(h)
            h = block_spade.res1(h, seg)
            h = block_spade.res2(h, seg)
        h = self.conv_last_spade(h)
        h = torch.tanh(h)
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


class Spade_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32, label_nc=37):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.label_nc = label_nc
        self.spade_norm1 = SPADEGroupNorm3D(in_channels, label_nc=self.label_nc)  #  Normalize(in_channels, norm_type, num_groups=num_groups)

        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)

        self.spade_norm2 = SPADEGroupNorm3D(in_channels, label_nc=self.label_nc)          #Normalize(in_channels, norm_type, num_groups=num_groups)

        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x, seg):
        h = x
        h = self.spade_norm1(h, seg)
        h = silu(h)
        h = self.conv1(h)
        h = self.spade_norm2(h, seg)
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
