#!/bin/bash
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --job-name=ptosto


module load python
conda activate of_dev

# python npt.py --lateral-size 512
python npt.py --lateral-size 2000 --hold-time-ps 50
