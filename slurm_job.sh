#!/bin/bash
#SBATCH --job-name=slurm
#SBATCH --account=dsl_jobs
#SBATCH --time=03:00
#SBATCH --output=slurm_logs/job%j.out

python src/main.py 