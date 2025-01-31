#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48

#SBATCH --gres=gpu:4

#SBATCH --time=23:00:00
#SBATCH --job-name=ptosto

 
module purge
module load anaconda3/2023.3
conda activate /home/pinchenx/data.gpfs/envs/jax
python npt.py

