"Adapted from https://github.com/SongweiGe/TATS"
import sys
sys.path.append('/misc/no_backups/d1502/medicaldiffusion')
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from vq_gan_3d.model import VQGAN, VQGAN_SPADE, VQVAE
from train.callbacks import ImageLogger, VideoLogger
from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict


@hydra.main(config_path='/misc/no_backups/d1502/medicaldiffusion/config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu/8, bs/4, base_lr))

    if cfg.dataset.name == 'SemanticMap' or cfg.model.name == 'vq_vae':
        model = VQVAE(cfg, val_dataloader=val_dataloader)
    elif cfg.model.name == 'vq_gan_spade':
        model = VQGAN_SPADE(cfg, val_dataloader=val_dataloader)
    else:
        model = VQGAN(cfg, val_dataloader=val_dataloader)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=40000, save_top_k=-1,
                     filename='{epoch}-{step}-20000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(batch_frequency=3000, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=4000, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            if os.path.exists(ckpt_folder):
                for fn in os.listdir(ckpt_folder):
                    if fn == 'latest_checkpoint.ckpt':
                        ckpt_file = 'latest_checkpoint_prev.ckpt'
                        os.rename(os.path.join(ckpt_folder, fn),
                                  os.path.join(ckpt_folder, ckpt_file))

            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)

    accelerator = None
    if cfg.model.gpus > 1:
        accelerator = 'ddp'

    trainer = pl.Trainer(
        gpus=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    run()
