#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=U-Net
#SBATCH --output=U-Net%j.%N.out
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
pyenv activate myenv #myenv for diffusion myenv38 for vqgan #myenv #venv
pip uninstall nvidia_cublas_cu11
nvcc --version
python -c "import torch; print(torch.__version__)"

# Run your python code

python /misc/no_backups/d1502/medicaldiffusion/train/train_unet.py