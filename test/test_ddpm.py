import sys
sys.path.append('/misc/no_backups/d1502/medicaldiffusion')
from ddpm import Unet3D, GaussianDiffusion, Trainer, Unet3D_SPADE, SemanticGaussianDiffusion, Semantic_Trainer
import hydra
from omegaconf import DictConfig, open_dict
from train.get_dataset import get_dataset
import torch
import os
from ddpm.unet import UNet
from torch.utils.data import DataLoader
# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py
from evaluation.metrics import Metrics

@hydra.main(config_path='/misc/no_backups/d1502/medicaldiffusion/config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)
    print(cfg.model.denoising_fn, "and", cfg.model.diffusion, 'are implemented')
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
        ).cuda()
    elif cfg.model.denoising_fn == 'Unet3D_SPADE':
        model = Unet3D_SPADE(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            label_nc=cfg.dataset.label_nc,
            segconv=cfg.model.segconv
        ).cuda()
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    if cfg.model.diffusion == 'SemanticGaussianDiffusion':
        model = SemanticGaussianDiffusion(
            model,
            vqgan_ckpt=cfg.model.vqgan_ckpt,
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            loss_type=cfg.model.loss_type,
            cond_scale=cfg.model.cond_scale
        ).cuda()
    elif cfg.model.diffusion == 'GaussianDiffusion':
        model = GaussianDiffusion(
            model,
            vqgan_ckpt=cfg.model.vqgan_ckpt,
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            loss_type=cfg.model.loss_type,
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.diffusion} doesn't exist")

    def preprocess_input(self, data):
        # move to GPU and change data types
        label = data['label'].cuda().long()
        img = data['image'].cuda().float()
        # create one-hot label map
        bs, _, t, h, w = label.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, t, h, w).zero_().cuda()
        input_semantics = input_label.scatter_(1, label, 1.0)

        return img, input_semantics

    def load_checkpoint(model, results_folder, **kwargs):
        all_paths = os.listdir(os.path.join(results_folder, 'checkpoints'))
        all_milestones = [int((p.split('.')[0]).split("-")[-1]) for p in all_paths if p.endswith('.pt')]
        assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
        milestone = max(all_milestones)

        data = torch.load(os.path.join(results_folder, 'checkpoints', f"sample-{milestone}.pt"))

        model.load_state_dict(data['model'], **kwargs)
        # model.load_state_dict(data['ema'], **kwargs)
        print("checkpoint is successful loaded")

        return model

    results_folder = "/misc/no_backups/d1502/medicaldiffusion/checkpoints/ddpm/AutoPET/output_with_segconv/checkpoints"
    metrics_folder = ""
    train_dataset, val_dataset, _ = get_dataset(cfg)
    model = load_checkpoint(model, results_folder)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    compute_matrics = True
    generate_gif = True
    if compute_matrics:
        metrics_computer = Metrics(metrics_folder, val_dl)
        metrics_computer.metrics_test(model)
    if generate_gif:
        model.eval()
        with torch.no_grad():
            for i, data_i in enumerate(val_dl):
                label_save = data_i['label']
                image, label = preprocess_input(data_i)
                generated = model.sample(cond=label)
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












if __name__ == '__main__':
    run()

    # wandb.finish()
    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)