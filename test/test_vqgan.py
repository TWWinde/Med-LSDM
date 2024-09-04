"Adapted from https://github.com/SongweiGe/TATS"
import sys
import torch
sys.path.append('/misc/no_backups/d1502/medicaldiffusion')

from evaluation.metrics_vq_gan import metrics
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from vq_gan_3d.model import VQGAN, VQGAN_SPADE, VQVAE
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict
from torchvision import transforms as T
import torch.nn.functional as F
from einops import rearrange
import io


@hydra.main(config_path='/misc/no_backups/d1502/medicaldiffusion/config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=cfg.model.num_workers)

    with open_dict(cfg):
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)

    if cfg.dataset.name == 'SemanticMap' or cfg.model.name == 'vq_vae':
        model = VQVAE(cfg, val_dataloader=val_dataloader)
    elif cfg.model.name == 'vq_gan_spade':
        model = VQGAN_SPADE(cfg, val_dataloader=val_dataloader)
    else:
        model = VQGAN(cfg, val_dataloader=val_dataloader)

    # load the most recent checkpoint file
    model = model.load_from_checkpoint(cfg.model.resume_from_checkpoint)
    model.eval()
    model.freeze()
    model.cuda()
    results_folder = os.path.join("/data/private/autoPET/medicaldiffusion_results/test_results/", cfg.model.name, cfg.dataset.name)
    os.makedirs(results_folder, exist_ok=True)
    with torch.no_grad():
        metrics_computer = metrics(results_folder, val_dataloader, num_classes=cfg.dataset.label_nc)
        if cfg.model.name == "vq_gan_3d":
            metrics_computer.metrics_test(model)
        elif cfg.model.name == "vq_gan_spade":
            metrics_computer.metrics_test_spade(model)
        elif cfg.model.name == "vq_vae":
            metrics_computer.metrics_test(model)

        for i, data_i in enumerate(val_dataloader):
            input = data_i['image']
            output = model(input, evaluation=True)

            input_list = F.pad(input, (2, 2, 2, 2))
            recon_list = F.pad(output, (2, 2, 2, 2))

            input_gif = rearrange(input_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
            recon_gif = rearrange(recon_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
            path_video = os.path.join(results_folder, 'video_results')
            os.makedirs(path_video, exist_ok=True)

            image_path = os.path.join(path_video, f'{i}_input.gif')
            label_path = os.path.join(path_video, f'{i}_recon.gif')
            video_tensor_to_gif(input_gif, image_path)
            video_tensor_to_gif(recon_gif, label_path)


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


if __name__ == '__main__':
    run()
