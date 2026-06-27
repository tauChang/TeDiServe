import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


NUM_GPUS = 4
RPS_ORDER = [7.5, 8.0, 8.5, 9.0]
SYSTEM_ORDER = ["infaas", "async_no_load_control", "async"]
SYSTEM_LABELS = {
    "infaas": "Fixed thresh.",
    "async_no_load_control": "Dynamic thresh.",
    "async": "Dynamic thresh.\n+ Load control",
}
COLORS = {
    "infaas": "#134686",
    "async_no_load_control": "#6BAED6",
    "async": "#FF4F0F",
}
METRICS = {
    "slo": {
        "ylabel": "SLO Attain. (%)",
        "ylim": (0.0, 1.05),
        "yticks": np.arange(0.0, 1.01, 0.25),
    },
    "acc": {
        "ylabel": "Accuracy (%)",
        "ylim": (0.7, 0.79),
        "yticks": np.arange(0.7, 0.791, 0.02),
    },
}

PERCENT_INTEGER_FORMATTER = FuncFormatter(lambda value, _: f"{value * 100:.0f}")


def format_rps_per_gpu_label(rps: float) -> str:
    return f"{rps / NUM_GPUS:.2f}".rstrip("0").rstrip(".")


def parse_summary(summary_path: Path):
    pattern = re.compile(r"^(infaas|async_no_load_control|async)_([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
    data = {metric: {system: [] for system in SYSTEM_ORDER} for metric in METRICS}

    with open(summary_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if not match:
                continue

            system, rps_str, slo_str, acc_str = match.groups()
            rps = float(rps_str)
            if rps not in RPS_ORDER:
                continue

            data["slo"][system].append((rps, float(slo_str)))
            data["acc"][system].append((rps, float(acc_str)))

    for metric in data.values():
        for system, values in metric.items():
            values.sort(key=lambda item: item[0])

    return data


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_summary(data, output_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.4), constrained_layout=True)
    x = np.arange(len(RPS_ORDER))
    bar_width = 0.26
    offsets = [-bar_width, 0.0, bar_width]

    legend_handles = []
    legend_labels = []

    for ax, (metric_key, metric_config) in zip(axes, METRICS.items()):
        for offset, system in zip(offsets, SYSTEM_ORDER):
            values_by_rps = {rps: value for rps, value in data[metric_key][system]}
            y_values = [values_by_rps[rps] for rps in RPS_ORDER]
            bars = ax.bar(
                x + offset,
                y_values,
                width=bar_width,
                color=COLORS[system],
                edgecolor="black",
                linewidth=1,
                label=SYSTEM_LABELS[system],
            )
            if metric_key == "slo":
                legend_handles.append(bars[0])
                legend_labels.append(SYSTEM_LABELS[system])

        ax.set_ylabel(metric_config["ylabel"], labelpad=0)
        ax.set_xlabel("RPS per GPU", labelpad=1)
        ax.set_xticks(x)
        ax.set_xticklabels([format_rps_per_gpu_label(rps) for rps in RPS_ORDER])
        ax.set_ylim(*metric_config["ylim"])
        ax.set_yticks(metric_config["yticks"])
        ax.yaxis.set_major_formatter(PERCENT_INTEGER_FORMATTER)
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", direction="in", length=1)

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1.0, 0.55),
        frameon=True,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.8,
        alignment="left"
    )

    legend.get_texts()[2].set_multialignment("left")
    legend.get_texts()[2].set_verticalalignment("baseline")

    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("final_summary.txt"),
        help="Path to the copied final_summary.txt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("conf_adjustment_ablation.png"),
        help="Output path for the ablation figure.",
    )
    args = parser.parse_args()

    data = parse_summary(args.input)
    plot_summary(data, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()