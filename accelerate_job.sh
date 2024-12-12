#!/bin/bash
#SBATCH --job-name=acceleratejob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40g
#SBATCH --gpus=4
#SBATCH --gres=gpumem:24g
#SBATCH --time=08:00:00
#SBATCH --output=accelerate_logs/job%j.out
#SBATCH --error=accelerate_logs/job_%j.err

source /cluster/home/ckeusch/miniconda3/bin/activate
conda activate dsl

accelerate launch --multi-gpu --mixed_precision=fp16 src/main.py 