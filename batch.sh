#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=3DVQGAN
#SBATCH --output=VQ-GAN%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse21
#SBATCH --qos=shortbatch
#SBATCH --partition=highperf
# SBATCH --gpus=rtx_a5000:1
# SBATCH --nodelist=linse19


# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate myenv38 #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"
export CUDA_LAUNCH_BLOCKING=1

# Run your python code


# autopet
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=autopet \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 \  #8
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

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

# mr_wo_mask
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=synthrad2023_wo_mask \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# mr_wo_mask
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=totalsegmentator_mri \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# duke_
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# duke_bspline
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_bspline' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 dataset.root_dir='/misc/data/private/autoPET/duke/mr_bspline' \

# duke_no_rescale
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_randomcrop' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-5 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384  \

# duke_only_t1_real
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_only_t1_real' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=1e-5 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=4 model.n_codes=16384  \
#dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real_rescale_crop' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real_rescale_crop'

# duke_only_t1_randomcrop
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_randomcrop_t1_real' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=1e-5 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=4 model.n_codes=16384  \
#dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real'

# duke_t1_all
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=duke \
model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_t1_all' model.precision=32 model.embedding_dim=8 \
model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384  \
dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real_all_rescale_crop' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real_all_rescale_crop'

# semantic map
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=semanticmap \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# vq_gan_spade  137077
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=autopet \
#model=vq_gan_spade model.gpus=1 model.default_root_dir_postfix='vq_gan_spade' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384

# vq_gan_spade
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=synthrad2023 \
#model=vq_gan_spade model.gpus=1 model.default_root_dir_postfix='vq_gan_spade' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384

# vq_vae seg_map autopet segmap
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=semanticmap \
#model=vq_vae model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.batch_size=2 model.n_codes=16384

# vq_vae seg_map synthrad mr
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=synthrad2023 \
#model=vq_vae model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.batch_size=2 model.n_codes=16384
