#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=3D_Diffusion
#SBATCH --output=Diffusion%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
# SBATCH --gpus=rtx_a5000:1
# SBATCH --nodelist=linse19
#SBATCH --nodelist=linse23
#SBATCH --qos=shortbatch
#SBATCH --partition=highperf


# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate myenv #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"

# Run your python code

# diffusion without condition
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output' \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/flair/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D  model.diffusion=GaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8]  model.batch_size=10 model.gpus=0

# diffusion with condition
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output' \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/flair/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=4 model.gpus=0

# diffusion with seggan condition 137083
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_seggan' dataset.label_nc=8 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/flair/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.seggan_ckpt="/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/SemanticMap/results/lightning_logs/version_136418/checkpoints/latest_checkpoint.ckpt" \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0

# diffusion without condition
#python train/train_ddpm.py model=ddpm dataset=synthrad2023_wo_mask model.results_folder_postfix="output' model.name=vq_gan_3d \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/SynthRAD2023_wo_mask/flair/lightning_logs/version_133984/checkpoints/latest_checkpoint.ckpt' \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D  model.diffusion=GaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8]  model.batch_size=10 model.gpus=0

# diffusion with segconv condition 137058
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.seggan_ckpt=0 model.spade_input_channel=37 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1

# diffusion with segconv 64 condition
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv_64out' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.seggan_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1

# diffusion with segconv 128 condition
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv_128out' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.seggan_ckpt=0 model.spade_input_channel=128 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1

# diffusion with vg_gan_SPADE
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_vq_gan_spade' dataset.label_nc=37 \
#model.vqgan_spade_ckpt='/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan_spade/AutoPET/vq_gan_spade/lightning_logs/version_137077/checkpoints/latest_checkpoint.ckpt' \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0

# diffusion with segconv 64 condition vq_gan_spade autopet
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_vq_spade_segconv_64out' dataset.label_nc=37 \
#model.vqgan_ckpt=0 \
#model.vqgan_spade_ckpt="/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan_spade/AutoPET/vq_gan_spade/lightning_logs/version_137357/checkpoints/latest_checkpoint.ckpt" \
#model.seggan_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion with segconv 64 condition with vlb loss
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_vlb_segconv_64out' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.seggan_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion with segconv 64 condition mr
python train/train_ddpm.py model=ddpm dataset=synthrad2023 model.results_folder_postfix='output_with_segconv_64out' dataset.label_nc=31 \
model.vqgan_ckpt='/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/SynthRAD2023/results/lightning_logs/version_133718/checkpoints/latest_checkpoint.ckpt' \
model.seggan_ckpt=0 model.spade_input_channel=64 \
model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1