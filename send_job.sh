#!/bin/bash

#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 2
#SBATCH --time=1440
#SBATCH --account=dl_jobs
#SBATCH --output=sbatched.out

python src/main.py user=tim_cluster
