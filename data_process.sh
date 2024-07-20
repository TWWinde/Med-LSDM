#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=data_process
#SBATCH --output=PROCESS%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-23:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --gpus=rtx_a5000:1
# SBATCH --qos=shortbatch
# SBATCH --partition=highperf

# Activate everything you need


module load cuda
pyenv activate myenv
pip uninstall nvidia_cublas_cu11

# Run your python code

python /no_backups/d1502/medicaldiffusion/dataset/data_process.py