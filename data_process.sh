#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=data_process
#SBATCH --output=PROCESS%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --gpus=rtx_a5000:1

# Activate everything you need

#conda activate /anaconda3/envs/myenv
module load cuda
pyenv activate venv
pip uninstall nvidia_cublas_cu11

# Run your python code

python /no_backups/d1502/medicaldiffusion/dataset/data_processing.py