import sys
sys.path.append('/misc/no_backups/d1502/medicaldiffusion')
from ddpm import Unet3D, GaussianDiffusion, Unet3D_SPADE, SemanticGaussianDiffusion
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
    print(cfg.model.denoising_fn, "and", cfg.model.diffusion, 'are implemented')
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
            label_nc=cfg.model.spade_input_channel if cfg.model.segconv == 1 else cfg.dataset.label_nc,
            segconv=cfg.model.segconv
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
            unet_model,
            vqgan_ckpt=None if cfg.model.vqgan_ckpt == 0 else cfg.model.vqgan_ckpt,
            vqgan_spade_ckpt=None if cfg.model.vqgan_spade_ckpt == 0 else cfg.model.vqgan_spade_ckpt,
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            # sampling_timesteps=cfg.model.sampling_timesteps,
            loss_type=cfg.model.loss_type,
            cond_scale=cfg.model.cond_scale
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

    results_folder = os.path.join("/data/private/autoPET/medicaldiffusion_results/checkpoints/", cfg.model.name, cfg.dataset.name, cfg.model.results_folder_postfix)
    os.makedirs(results_folder, exist_ok=True)
    _, val_dataset, _ = get_dataset(cfg)
    checkpoint_folder = os.path.join("/misc/no_backups/d1502/medicaldiffusion/checkpoints", cfg.model.name, cfg.dataset.name, cfg.model.results_folder_postfix)
    diffusion_model = load_checkpoint(diffusion, checkpoint_folder)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    compute_matrics = True
    generate_gif = False
    if compute_matrics:
        metrics_computer = Metrics(results_folder, val_dl)
        metrics_computer.metrics_test(diffusion_model)
    if generate_gif:
        model.eval()
        with torch.no_grad():
            for i, data_i in enumerate(val_dl):
                label_save = data_i['label']
                image, label = preprocess_input(data_i)
                generated = diffusion_model.sample(cond=label)

                all_videos_list = F.pad(generated, (2, 2, 2, 2))
                all_label_list = F.pad(label_save, (2, 2, 2, 2))
                all_image_list = F.pad(image, (2, 2, 2, 2))

                sample_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                label_gif = rearrange(all_label_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                image_gif = rearrange(all_image_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                path_video = os.path.join(results_folder, 'video_results')
                os.makedirs(path_video, exist_ok=True)

                sample_path = os.path.join(path_video, f'{i}_sample.gif')
                image_path = os.path.join(path_video, f'{i}_image.gif')
                label_path = os.path.join(path_video, f'{i}_label.gif')
                video_tensor_to_gif(sample_gif, sample_path)
                video_tensor_to_gif(image_gif, image_path)
                video_tensor_to_gif(label_gif, label_path)


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