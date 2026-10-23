#!/bin/bash
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --job-name=sh8_sd3

module load python
conda activate of_dev


cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

TEMPERATURE=250
SURFACE_SHELL_THICKNESS_CELLS=8
SEED=0
size=256

SAMPLES_STEPS=500001
SAMPLES_DUMP_INTERVAL=50000
SURFACE_K2_SHIFT=5.502

diameter=$((size - 10))
radius=$(python -c "print(0.5 * ${diameter})")
center=$(python -c "print(0.5 * (${size} - 1))")

echo "T=${TEMPERATURE} D=${diameter} L=${size}"
OUTPUT_DIR="T${TEMPERATURE}D${diameter}L${size}_shell${SURFACE_SHELL_THICKNESS_CELLS}_seed${SEED}"
echo "Output directory: ${OUTPUT_DIR}"
python 01.npt.py \
    --temperature "${TEMPERATURE}" \
    --radius "${radius}" \
    --size "${size}" \
    --center "${center}" "${center}" "${center}" \
    --sampling-steps "${SAMPLES_STEPS}" \
    --sample-dump-interval "${SAMPLES_DUMP_INTERVAL}" \
    --surface-shell-thickness-cells "${SURFACE_SHELL_THICKNESS_CELLS}" \
    --surface-k2-shift "${SURFACE_K2_SHIFT}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}"
python 02.analyze.py --output-dir "${OUTPUT_DIR}"
python 03.animation.py --input-dir "${OUTPUT_DIR}"
