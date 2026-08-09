"""Plot single-node simulation efficiency and peak GPU memory use."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV written by benchmark.py.")
    parser.add_argument("--output", type=Path, default=Path("benchmark.png"))
    parser.add_argument(
        "--summary", type=Path, default=Path("benchmark_summary.csv")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding="utf-8") as stream:
        rows = sorted(
            csv.DictReader(stream),
            key=lambda row: (int(row["lattice_size"]), int(row["gpu_count"])),
        )

    summary_fields = list(rows[0]) + ["speedup", "parallel_efficiency_percent"]
    summary_rows = []
    for size in (120, 240, 480):
        size_rows = [row for row in rows if int(row["lattice_size"]) == size]
        throughput = np.asarray(
            [float(row["steps_per_second"]) for row in size_rows]
        )
        gpu_count = np.asarray([int(row["gpu_count"]) for row in size_rows])
        speedup = throughput / throughput[0]
        efficiency = 100 * speedup / gpu_count
        for row, run_speedup, run_efficiency in zip(
            size_rows, speedup, efficiency
        ):
            summary_rows.append(
                row
                | {
                    "speedup": run_speedup,
                    "parallel_efficiency_percent": run_efficiency,
                }
            )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=summary_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for size, color in zip((120, 240, 480), colors):
        size_rows = [
            row for row in summary_rows if int(row["lattice_size"]) == size
        ]
        gpu_count = np.asarray([int(row["gpu_count"]) for row in size_rows])
        efficiency = np.asarray(
            [float(row["parallel_efficiency_percent"]) for row in size_rows]
        )
        memory_per_gpu = np.asarray(
            [float(row["peak_memory_per_gpu_gib"]) for row in size_rows]
        )
        axes[0].plot(gpu_count, efficiency, "o-", color=color, label=f"L={size}")
        axes[1].plot(
            gpu_count, memory_per_gpu, "o-", color=color, label=f"L={size}"
        )

    axes[0].axhline(100, color="0.45", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Parallel efficiency [%]")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("Peak JAX memory / GPU [GiB]")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.15, 25)
    for axis in axes:
        axis.set_xlabel("Number of A100 GPUs")
        axis.set_xticks((1, 2, 3, 4))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(bottom=0)

    fig.suptitle("BaTiO$_3$ NPT at 270 K with Ewald: single-node scaling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
