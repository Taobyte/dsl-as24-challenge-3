#!/bin/bash
#SBATCH --job-name=slurm
#SBATCH --account=dsl_jobs
#SBATCH --time=05:00
#SBATCH --output=slurm_logs/job%j.out

python src/main.py 