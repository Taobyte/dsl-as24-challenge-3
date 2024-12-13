#!/bin/bash
#SBATCH --job-name=slurm
#SBATCH --account=dsl
#SBATCH --gpus=1
#SBATCH --time=02:00
#SBATCH --output=slurm_logs/job%j.out

python src/main.py 