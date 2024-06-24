#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=3D_Diffusion
#SBATCH --output=Diffusion%j.%N.out
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
pyenv activate myenv #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"

# Run your python code

# diffusion
python train/train_ddpm.py model=ddpm dataset=autopet model.results_folder_postfix='output' \
model.vqgan_ckpt='/no_backups/d1502/medicaldiffusion/checkpoints/vq_gan/AutoPET/flair/lightning_logs/version_133784/checkpoints/latest_checkpoint.ckpt' \
model.diffusion_img_size=64 model.diffusion_depth_size=64 \
model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=0