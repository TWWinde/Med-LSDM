#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=3DVQGAN
#SBATCH --output=VQ-GAN%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --qos=shortbatch
#SBATCH --partition=highperf


# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate myenv38 #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"

# Run your python code

# autopet
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=autopet \
model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 \
model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# autopet_wo_artifacts
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=autopet_wo_artifacts \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# mr_wo_artifacts
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=synthrad2023 \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# diffusion
#python train/train_ddpm.py model=ddpm dataset=brats model.results_folder_postfix='flair' \
#model.vqgan_ckpt='/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/BRATS/flair/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt' \
#model.diffusion_img_size=32 model.diffusion_depth_size=32 \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=1