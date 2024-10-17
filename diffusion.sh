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
#SBATCH --nodelist=linse21
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

# diffusion with vqvae condition 137083
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_vqvae' dataset.label_nc=8 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/flair/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt="/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/SemanticMap/results/lightning_logs/version_136418/checkpoints/latest_checkpoint.ckpt" \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0

# diffusion without condition
#python train/train_ddpm.py model=ddpm dataset=synthrad2023_wo_mask model.results_folder_postfix="output' model.name=vq_gan_3d \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/SynthRAD2023_wo_mask/flair/lightning_logs/version_133984/checkpoints/latest_checkpoint.ckpt' \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D  model.diffusion=GaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8]  model.batch_size=10 model.gpus=0

# diffusion with segconv condition 37 137058
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=37 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1

# diffusion with segconv 64 condition
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv_64out' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1

# diffusion with segconv 128 condition
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv_128out' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=128 \
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
#model.vqvae_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion with segconv 64 condition mr
#python train/train_ddpm.py model=ddpm dataset=synthrad2023 model.results_folder_postfix='output_with_segconv_64out' dataset.label_nc=31 \
#model.vqgan_ckpt='/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/SynthRAD2023/results/lightning_logs/version_133718/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion with vqvae condition autopet
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_vqvae' dataset.label_nc=37 \
#model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt='/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_vae/SemanticMap/results/lightning_logs/version_139795/checkpoints/latest_checkpoint.ckpt' \
#model.spade_input_channel=37 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=0 model.load_milestone=0

# diffusion with segconv 64 condition vq_gan_spade mr
#python train/train_ddpm.py model=ddpm dataset=synthrad2023 model.results_folder_postfix='output_mr_vq_spade_segconv_64out' dataset.label_nc=31 \
#model.vqgan_ckpt=0 \
#model.vqgan_spade_ckpt="/misc/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan_spade/SynthRAD2023/vq_gan_spade/lightning_logs/version_137360/checkpoints/latest_checkpoint.ckpt" \
#model.vqvae_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion with segconv 64 condition duke
#python train/train_ddpm.py model=ddpm dataset=duke model.results_folder_postfix='results_randomcrop' dataset.label_nc=3 \
#model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_randomcrop/lightning_logs/version_141449/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=64 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion autopet with segconv 64 condition add seg to noise
#python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output__addsegtonoise_segconv_64' dataset.label_nc=37 \
#model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/AutoPET/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=64 model.add_seg_to_noise=1 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion synthrad with segconv 64 condition
#python train/train_ddpm.py model=ddpm dataset=synthrad2023 model.results_folder_postfix='output_mr_segconv_64_tanh' dataset.label_nc=31 \
#model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/SynthRAD2023/mr_final/lightning_logs/version_144347/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=64 model.add_seg_to_noise=0 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion with segconv 64 condition duke
#python train/train_ddpm.py model=ddpm dataset=duke model.results_folder_postfix='results_duke_final_32' dataset.label_nc=3 \
#model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_t1_all_tanh/lightning_logs/version_144222/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=32 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0 model.segconv=1 model.load_milestone=0 \
#dataset.root_dir='/data/private/autoPET/duke/final_labeled_mr' dataset.val_dir='/data/private/autoPET/duke/final_labeled_mr'

# diffusion with segconv 8 condition duke
python train/train_ddpm.py model=ddpm dataset=duke model.results_folder_postfix='results_duke_final_8' dataset.label_nc=3 \
model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_t1_all_tanh/lightning_logs/version_144222/checkpoints/latest_checkpoint.ckpt' \
model.vqvae_ckpt=0 model.spade_input_channel=8 \
model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0 model.segconv=1 model.load_milestone=0 \
dataset.root_dir='/data/private/autoPET/duke/final_labeled_mr_test' dataset.val_dir='/data/private/autoPET/duke/final_labeled_mr_test'

# diffusion with segconv 3 condition duke
#python train/train_ddpm.py model=ddpm dataset=duke model.results_folder_postfix='results_duke_final_3' dataset.label_nc=3 \
#model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_t1_all_tanh/lightning_logs/version_144222/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=3 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0 model.segconv=1 model.load_milestone=0 \
#dataset.root_dir='/data/private/autoPET/duke/final_labeled_mr' dataset.val_dir='/data/private/autoPET/duke/final_labeled_mr'

# diffusion with segconv 64 condition duke T1_MR_real
#python train/train_ddpm.py model=ddpm dataset=duke model.results_folder_postfix='results_duke_real_randomcrop_8' dataset.label_nc=3 \
#model.vqgan_ckpt='/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan/DUKE/results_randomcrop_t1_real_tanh/lightning_logs/version_144594/checkpoints/latest_checkpoint.ckpt' \
#model.vqvae_ckpt=0 model.spade_input_channel=8 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0 model.segconv=1 model.load_milestone=0 \
#dataset.root_dir='/misc/data/private/autoPET/duke/T1_MR_real' dataset.val_dir='/misc/data/private/autoPET/duke/T1_MR_real'

# diffusion with segconv 8 condition vq_gan_spade duke
#python train/train_ddpm.py model=ddpm dataset=duke model.results_folder_postfix='results_duke_vq_spade_segconv_8out' dataset.label_nc=3 \
#model.vqgan_ckpt=0 \
#model.vqgan_spade_ckpt="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan_spade/DUKE/vq_gan_spade_tanh_duke/lightning_logs/version_145400/checkpoints/latest_checkpoint.ckpt" \
#model.vqvae_ckpt=0 model.spade_input_channel=8 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1 model.load_milestone=0

# diffusion vq_gan_spade synthrad2023
#python train/train_ddpm.py model=ddpm dataset=synthrad2023 model.results_folder_postfix='results_synthrad_spade_tanh_segconv_31out' dataset.label_nc=31 \
#model.vqgan_ckpt=0 \
#model.vqgan_spade_ckpt="/data/private/autoPET/medicaldiffusion_results/results/checkpoints/vq_gan_spade/SynthRAD2023/vq_gan_spade_mr_tanh/lightning_logs/version_146008/checkpoints/latest_checkpoint.ckpt" \
#model.vqvae_ckpt=0 model.spade_input_channel=31 \
#model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
#model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=0 model.segconv=1 model.load_milestone=0
