#!/bin/bash
#SBATCH --account=m5218
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --job-name=ptosto


module load python
conda activate of_dev

srun --ntasks=1 --gpus-per-task=1 python npt.py
