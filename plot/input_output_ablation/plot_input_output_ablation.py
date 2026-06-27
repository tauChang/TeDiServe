import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NUM_GPUS = 4
COLUMN_ORDER = ["2shot", "5shot", "8shot", "out64", "out256", "out1024"]
COLUMN_GROUPS = {
    "input": ["2shot", "5shot", "8shot"],
    "output": ["out64", "out256", "out1024"],
}
X_TICK_STEPS = {
    "out1024": 0.05,
}
X_AXIS_LIMITS = {
    "out1024": (0.37, 0.475),
}
COLUMN_LABELS = {
    "2shot": "2-shot",
    "5shot": "5-shot",
    "8shot": "8-shot",
    "out64": "Out Len. 64",
    "out256": "Out Len. 256",
    "out1024": "Out Len. 1024",
}
SYSTEM_LABELS = {
    "infaas": "INFaaS",
    "tedi": "TeDiServe",
}
COLORS = {
    "INFaaS": "#134686",
    "TeDiServe": "#FF4F0F",
}
MARKERS = {
    "INFaaS": "d",
    "TeDiServe": "o",
}
ZORDER = {
    "INFaaS": 2,
    "TeDiServe": 3,
}


def parse_rps_token(rps_token: str) -> float:
    return float(rps_token.replace("_", "."))


def parse_summary(summary_path: Path):
    pattern = re.compile(
        r"^(infaas|tedi)_(2shot|5shot|8shot|out64|out256|out1024)_rps([0-9_]+)\s*\|"
        r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    )

    data = {
        column: {
            SYSTEM_LABELS[system]: {"rps": [], "slo": [], "acc": []}
            for system in SYSTEM_LABELS
        }
        for column in COLUMN_ORDER
    }

    with open(summary_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if not match:
                continue

            system_key, column_key, rps_token, slo_str, acc_str = match.groups()
            system = SYSTEM_LABELS[system_key]
            rps_per_gpu = parse_rps_token(rps_token) / NUM_GPUS

            data[column_key][system]["rps"].append(rps_per_gpu)
            data[column_key][system]["slo"].append(float(slo_str))
            data[column_key][system]["acc"].append(float(acc_str))

    return data


def sorted_series(series):
    if not series["rps"]:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(series["rps"])
    rps = np.array(series["rps"])[order]
    slo = np.array(series["slo"])[order]
    acc = np.array(series["acc"])[order]
    return rps, slo, acc


def compute_accuracy_ylim(data):
    values = []
    for column in COLUMN_ORDER:
        for system in SYSTEM_LABELS.values():
            values.extend(data[column][system]["acc"])

    if not values:
        return 0.0, 1.0

    lower = min(values)
    upper = max(values)
    padding = max((upper - lower) * 0.1, 0.01)
    return lower - padding, upper + padding


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def compute_ticks(values, step):
    if len(values) == 0:
        return None

    lower = np.floor(min(values) / step) * step
    upper = np.ceil(max(values) / step) * step
    return np.arange(lower, upper + 0.001, step)


def plot_group(data, columns, output_path: Path, acc_ylim):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })

    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(3.4, 1.8),
        sharey="row",
        constrained_layout=True,
    )
    

    if len(columns) == 1:
        axes = np.array(axes).reshape(2, 1)

    legend_handles = None
    legend_labels = None

    for col_idx, column in enumerate(columns):
        ax_slo = axes[0, col_idx]
        ax_acc = axes[1, col_idx]
        column_rps_values = []

        for system in ["TeDiServe", "INFaaS"]:
            rps, slo, acc = sorted_series(data[column][system])
            if len(rps) == 0:
                continue

            column_rps_values.extend(rps.tolist())

            line, = ax_slo.plot(
                rps,
                slo,
                color=COLORS[system],
                marker=MARKERS[system],
                label=system,
                zorder=ZORDER[system],
            )
            ax_acc.plot(
                rps,
                acc,
                color=COLORS[system],
                marker=MARKERS[system],
                label=system,
                zorder=ZORDER[system],
            )

            if legend_handles is None:
                legend_handles = [line]
                legend_labels = [system]
            elif system not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(system)

        ax_slo.set_title(COLUMN_LABELS[column])
        ax_slo.grid(True, alpha=0.4)
        ax_acc.grid(True, alpha=0.4)
        ax_slo.set_ylim(0.0, 1.05)
        ax_slo.set_yticks(np.arange(0.0, 1.01, 0.25))
        ax_acc.set_ylim(*acc_ylim)
        ax_acc.set_yticks(np.arange(0.60, 0.81, 0.05))

        if col_idx == 0:
            ax_slo.set_ylabel("SLO Attain.")
            ax_acc.set_ylabel("Accuracy")

        ax_acc.set_xlabel("RPS per GPU")
        tick_step = X_TICK_STEPS.get(column, 0.5)
        xticks = compute_ticks(column_rps_values, tick_step)
        if xticks is not None:
            ax_slo.set_xticks(xticks)
            ax_acc.set_xticks(xticks)
        if column in X_AXIS_LIMITS:
            x_min, x_max = X_AXIS_LIMITS[column]
            ax_slo.set_xlim(x_min, x_max)
            ax_acc.set_xlim(x_min, x_max)
        ax_slo.sharex(ax_acc)
        plt.setp(ax_slo.get_xticklabels(), visible=False)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.58, 1.12),
            frameon=True,
            handlelength=1.5,
            columnspacing=1.0,
        )

    save_figure(fig, output_path)


def plot_summary(data, output_path: Path):
    acc_ylim = compute_accuracy_ylim(data)

    stem = output_path.stem
    suffix = output_path.suffix

    input_output_path = output_path.with_name(f"{stem}_input{suffix}")
    output_output_path = output_path.with_name(f"{stem}_output{suffix}")

    plot_group(data, COLUMN_GROUPS["input"], input_output_path, acc_ylim)
    plot_group(data, COLUMN_GROUPS["output"], output_output_path, acc_ylim)

    return [input_output_path, output_output_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./input_output_ablation.txt"),
        help="Path to the ablation summary text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./io_ablation.png"),
        help="Path to the output plot image.",
    )
    args = parser.parse_args()

    data = parse_summary(args.input)
    outputs = plot_summary(data, args.output)
    for output in outputs:
        print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()