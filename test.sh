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

# diffusion with segconv condition
python /test/test_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output_with_segconv' dataset.label_nc=37 \
model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET1/results/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
model.seggan_ckpt=0 model.spade_input_channel=37 \
model.diffusion_img_size=64 model.diffusion_depth_size=8 model.denoising_fn=Unet3D_SPADE model.diffusion=SemanticGaussianDiffusion \
model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=1 model.gpus=0 model.segconv=1

