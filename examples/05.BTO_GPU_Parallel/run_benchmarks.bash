#!/bin/bash
set -euo pipefail

module load python
conda activate of_dev

EXAMPLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$EXAMPLE_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
BTO_STEPS=${BTO_STEPS:-100}
BTO_WARMUP_STEPS=${BTO_WARMUP_STEPS:-3}
BTO_REPEATS=${BTO_REPEATS:-3}
RUN_ID=${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=${BTO_RESULT_DIR:-"results/${RUN_ID}"}
RESULTS_CSV="$RESULT_DIR/benchmark_results.csv"

mkdir -p "$RESULT_DIR"

for size in 120 240 480; do
    for gpu_count in 1 2 3 4; do
        srun \
            --nodes=1 \
            --ntasks=1 \
            --gpus-per-task="$gpu_count" \
            --exact \
            --kill-on-bad-exit=1 \
            python benchmark.py \
                --size "$size" \
                --steps "$BTO_STEPS" \
                --warmup-steps "$BTO_WARMUP_STEPS" \
                --repeats "$BTO_REPEATS" \
                --output "$RESULTS_CSV"
    done
done

python plot_benchmarks.py "$RESULTS_CSV" \
    --summary "$RESULT_DIR/benchmark_summary.csv" \
    --output "$RESULT_DIR/benchmark.png"

echo "Results: $RESULT_DIR"
