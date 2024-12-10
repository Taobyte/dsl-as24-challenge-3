#!/bin/bash

#SBATCH --job-name=accelerate_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40g
#SBATCH --gpus=4
#SBATCH --gres=gpumem:24g
#SBATCH --time=04:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err


# Activate your conda environment (if using)
source /cluster/home/ckeusch/miniconda3/bin/activate
conda activate CDiffSD

# Run the command add : 
accelerate launch --multi-gpu --mixed_precision=fp16 src/main.py