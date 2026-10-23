#!/bin/bash
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --job-name=smo

module load python
conda activate of_dev

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

CUDA_VISIBLE_DEVICES=0 python run_coupled.py --size 64 --equilibration-steps 5000 --sampling-steps 50000 &
CUDA_VISIBLE_DEVICES=1 python run_frozen.py --size 64 --equilibration-steps 5000 --sampling-steps 50000 &
wait
echo "Finished."
 