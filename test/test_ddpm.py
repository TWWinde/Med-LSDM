import sys

import numpy as np

sys.path.append('/misc/no_backups/d1502/medicaldiffusion')
from ddpm import Unet3D, GaussianDiffusion, Unet3D_SPADE, SemanticGaussianDiffusion
from vq_gan_3d.model import VQVAE
import hydra
from omegaconf import DictConfig, open_dict
from train.get_dataset import get_dataset
import torch
import os
from ddpm.unet import UNet
from torch.utils.data import DataLoader
# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py
from evaluation.metrics import Metrics
from torchvision import transforms as T
import torch.nn.functional as F
from einops import rearrange


@hydra.main(config_path='/misc/no_backups/d1502/medicaldiffusion/config', config_name='base_cfg', version_base=None)
def inference(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)
    print("Task name:", cfg.model.results_folder_postfix, cfg.model.denoising_fn, "and", cfg.model.diffusion,
          'are implemented')
    if cfg.model.denoising_fn == 'Unet3D':
        unet_model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
        ).cuda()
    elif cfg.model.denoising_fn == 'Unet3D_SPADE':
        unet_model = Unet3D_SPADE(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            label_nc=cfg.dataset.label_nc,
            spade_input_nc=cfg.model.spade_input_channel if cfg.model.segconv != 0 else None,
            segconv=False if cfg.model.segconv == 0 else True,
            vqvae=False if cfg.model.vqvae_ckpt == 0 else cfg.model.vqvae_ckpt,
            add_seg_to_noise=False if cfg.model.add_seg_to_noise == 0 else True,
        ).cuda()
    elif cfg.model.denoising_fn == 'UNet':
        unet_model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    if cfg.model.diffusion == 'SemanticGaussianDiffusion':
        diffusion = SemanticGaussianDiffusion(
            cfg,
            unet_model,
            vqgan_ckpt=None if cfg.model.vqgan_ckpt == 0 else cfg.model.vqgan_ckpt,
            vqgan_spade_ckpt=None if cfg.model.vqgan_spade_ckpt == 0 else cfg.model.vqgan_spade_ckpt,
            vqvae_ckpt=None if cfg.model.vqvae_ckpt == 0 else cfg.model.vqvae_ckpt,
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            # sampling_timesteps=cfg.model.sampling_timesteps,
            loss_type=cfg.model.loss_type,
            cond_scale=cfg.model.cond_scale,
            add_seg_to_noise=False if cfg.model.add_seg_to_noise == 0 else True,
            # objective=cfg.objective
        ).cuda()
    elif cfg.model.diffusion == 'GaussianDiffusion':
        diffusion = GaussianDiffusion(
            unet_model,
            vqgan_ckpt=cfg.model.vqgan_ckpt,
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            # sampling_timesteps=cfg.model.sampling_timesteps,
            loss_type=cfg.model.loss_type,
            # objective=cfg.objective
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.diffusion} doesn't exist")

    def preprocess_input(data):
        # move to GPU and change data types
        label = data['label'].cuda().long()
        img = data['image'].cuda().float()
        # create one-hot label map
        bs, _, t, h, w = label.size()
        nc = cfg.dataset.label_nc
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label, 1.0)

        return img, input_semantics

    def load_checkpoint(model, results_folder, **kwargs):
        all_paths = os.listdir(os.path.join(results_folder, 'checkpoints'))
        all_milestones = [int((p.split('.')[0]).split("-")[-1]) for p in all_paths if p.endswith('.pt')]
        assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
        milestone = max(all_milestones)
        data = torch.load(os.path.join(results_folder, 'checkpoints', f"sample-{milestone}.pt"))
        print('load checkpoint: ', os.path.join(results_folder, 'checkpoints', f"sample-{milestone}.pt"))

        # model.load_state_dict(data['model'], **kwargs)
        model.load_state_dict(data['ema'], **kwargs)
        print("checkpoint is successful loaded")

        return model

    results_folder = os.path.join("/data/private/autoPET/medicaldiffusion_results/test_results/", cfg.model.name, cfg.dataset.name, cfg.model.results_folder_postfix)
    os.makedirs(results_folder, exist_ok=True)
    _, val_dataset, _ = get_dataset(cfg)
    checkpoint_folder = os.path.join("/data/private/autoPET/medicaldiffusion_results/results/checkpoints", cfg.model.name, cfg.dataset.name, cfg.model.results_folder_postfix)
    diffusion_model = load_checkpoint(diffusion, checkpoint_folder)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    if cfg.model.vqvae_ckpt != 0:
        vqvae = VQVAE.load_from_checkpoint(cfg.model.vqvae_ckpt).cuda()
        vqvae.eval()
        print('vqvae is implemented')
    else:
        vqvae = None

    compute_matrics = True
    generate_npy = False
    if compute_matrics:
        metrics_computer = Metrics(results_folder, val_dl, num_classes=cfg.dataset.label_nc)
        metrics_computer.metrics_test(diffusion_model, encoder=vqvae)
    if generate_npy:
        diffusion_model.eval()
        with torch.no_grad():
            for i, data_i in enumerate(val_dl):
                label_save = data_i["label"]
                image, seg = preprocess_input(data_i)

                generated = diffusion_model.sample(cond=seg, get_middle_process=False)
                #input1 = (generated + 1) / 2
                #input2 = (image + 1) / 2
                generated_np = generated.cpu().numpy()
                image_np = image.cpu().numpy()
                label_np = label_save.cpu().numpy()

                path_video = os.path.join(results_folder, 'video_results')
                os.makedirs(path_video, exist_ok=True)

                sample_np_path = os.path.join(path_video, 'fake', f'{i}_sample.npy')
                image_np_path = os.path.join(path_video, 'real', f'{i}_image.npy')
                label_np_path = os.path.join(path_video, 'label', f'{i}_label.npy')
                os.makedirs(os.path.join(path_video, 'fake'), exist_ok=True)
                os.makedirs(os.path.join(path_video, 'real'), exist_ok=True)
                os.makedirs(os.path.join(path_video, 'label'), exist_ok=True)
                np.save(sample_np_path, generated_np, allow_pickle=True, fix_imports=True)
                np.save(image_np_path, image_np, allow_pickle=True, fix_imports=True)
                np.save(label_np_path, label_np, allow_pickle=True, fix_imports=True)

                if i == 1000:
                    print("finished")
                    break


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


if __name__ == '__main__':
    inference()

    # wandb.finish()
    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)