#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=test_vqgan
#SBATCH --output=test_vqgan%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --gpus=rtx_a5000:1
# SBATCH --nodelist=linse19
# SBATCH --nodelist=linse21
# SBATCH --qos=shortbatch
# SBATCH --partition=highperf


# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate myenv #myenv for diffusion myenv38 for vqgan #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"

# Run your python code

# vq_gan autopet
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=autopet \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt" \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_autopet' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# vq_gan_spade autopet
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=autopet \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan_spade/AutoPET/vq_gan_spade/lightning_logs/version_137357/checkpoints/latest_checkpoint.ckpt" \
#model=vq_gan_spade model.gpus=1 model.default_root_dir_postfix='vq_gan_spade_autopet' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384

# vq_gan mr
PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=synthrad2023 \
model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/SynthRAD2023/results/lightning_logs/version_133718/checkpoints/latest_checkpoint.ckpt" \
model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_vqgan_mr' model.precision=16 model.embedding_dim=8 \
model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# vq_gan_spade mr
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=synthrad2023 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan_spade/SynthRAD2023/vq_gan_spade/lightning_logs/version_137360/checkpoints/latest_checkpoint.ckpt" \
#model=vq_gan_spade model.gpus=1 model.default_root_dir_postfix='results_vq_gan_spade_mr' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# vq_gan mr totalsegmentator
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=totalsegmentator_mri \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/TotalSegmentator_mri/results/lightning_logs/version_137427/checkpoints/latest_checkpoint.ckpt" \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# vq_gan mr totalsegmentator fine tuned with mr
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=totalsegmentator_mri \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/TotalSegmentator_mri/results/lightning_logs/version_139199/checkpoints/latest_checkpoint.ckpt" \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# vq_vae synthrad mr
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=synthrad2023 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_vae/SynthRAD2023/results/lightning_logs/version_139999/checkpoints/latest_checkpoint.ckpt" \
#model=vq_vae model.gpus=1 model.default_root_dir_postfix='results' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.batch_size=2 model.n_codes=16384

#PL_TORCH_DISTRIBUTED_BACKEND=gloo python test/test_vqgan.py dataset=duke \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_t1_all/lightning_logs/version_143921/checkpoints/latest_checkpoint_prev.ckpt" \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_t1_all' model.precision=32 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384  \
#dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real_all_rescale_crop' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real_all_rescale_crop'

# mr_wo_artifacts
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/d1502/medicaldiffusion/test/test_vqgan.py dataset=synthrad2023 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/SynthRAD2023/mr_final/lightning_logs/version_144347/checkpoints/latest_checkpoint.ckpt" \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='mr_final' model.precision=16 model.embedding_dim=8 \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=4 model.n_codes=16384

# duke_t1_all_tanh
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/d1502/medicaldiffusion/test/test_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_t1_all_tanh' model.precision=32 model.embedding_dim=8 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_t1_all_tanh/lightning_logs/version_144222/checkpoints/latest_checkpoint.ckpt" \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384  \
#dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real_all_rescale_crop' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real_all_rescale_crop'

# duke_t1_tanh
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/d1502/medicaldiffusion/test/test_vqgan.py dataset=duke \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='results_t1_real_tanh' model.precision=32 model.embedding_dim=8 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_t1_real_tanh/lightning_logs/version_144592/checkpoints/latest_checkpoint.ckpt" \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384  \
#dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real_rescale_crop' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real_rescale_crop'

# autopet_tanh
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/d1502/medicaldiffusion/test/test_vqgan.py dataset=autopet \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='autopet_tanh' model.precision=16 model.embedding_dim=8 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/AutoPET/autopet_tanh/lightning_logs/version_144368/checkpoints/latest_checkpoint.ckpt" \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384

# synthrad_tanh
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/d1502/medicaldiffusion/test/test_vqgan.py dataset=synthrad2023 \
#model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='mr_final_tanh' model.precision=16 model.embedding_dim=8 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/SynthRAD2023/mr_final/lightning_logs/version_144347/checkpoints/latest_checkpoint.ckpt" \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=4 model.n_codes=16384

# vq_gan_spade duke
#PL_TORCH_DISTRIBUTED_BACKEND=gloo python /misc/no_backups/d1502/medicaldiffusion/test/test_vqgan.py dataset=duke \
#model=vq_gan_spade model.gpus=1 model.default_root_dir_postfix='vq_gan_spade_duke' model.precision=16 model.embedding_dim=8 \
#model.resume_from_checkpoint="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan_spade/DUKE/vq_gan_spade_tanh_duke/lightning_logs/version_145400/checkpoints/latest_checkpoint.ckpt" \
#model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=1e-5 \
#model.discriminator_iter_start=1000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 \
#model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384