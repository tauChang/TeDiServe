import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
NUM_GPUS = 4
SELECTED_COLUMNS = ["2shot", "8shot", "out64", "out1024"]
SLO_Y_LIMITS = (0, 100)
SLO_Y_TICKS = [0, 25, 50, 75, 100]
ACC_Y_LIMITS = (58, 81)
ACC_Y_TICKS = [60, 65, 70, 75, 80]
X_AXIS_CONFIG = {
    "2shot": {
        "limits": (1.94, 3.05),
        "ticks": [2.0, 2.5, 3.0],
    },
    "8shot": {
        "limits": (0.43, 1.57),
        "ticks": [0.5, 1.0, 1.5],
    },
    "out64": {
        "limits": (5.45, 6.55),
        "ticks": [5.5, 6.0, 6.5],
    },
    "out1024": {
        "limits": (0.37, 0.475),
        "ticks": [0.40, 0.45],
    },
}
COLUMN_LABELS = {
    "2shot": "2-shot",
    "8shot": "8-shot",
    "out64": "Output 64",
    "out1024": "Output 1024",
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
LEFT_YLABEL_X = -0.33
COLUMN_GROUP_GAP = 0.018


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
        for column in SELECTED_COLUMNS
    }

    with open(summary_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if not match:
                continue

            system_key, column_key, rps_token, slo_str, acc_str = match.groups()
            if column_key not in data:
                continue

            system = SYSTEM_LABELS[system_key]
            rps_per_gpu = parse_rps_token(rps_token) / NUM_GPUS
            slo_pct = float(slo_str) * 100.0
            acc_pct = float(acc_str) * 100.0

            data[column_key][system]["rps"].append(rps_per_gpu)
            data[column_key][system]["slo"].append(slo_pct)
            data[column_key][system]["acc"].append(acc_pct)

    return data


def sorted_series(series):
    if not series["rps"]:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(series["rps"])
    rps = np.array(series["rps"])[order]
    slo = np.array(series["slo"])[order]
    acc = np.array(series["acc"])[order]
    return rps, slo, acc


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def plot_summary(data, output_path: Path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
        }
    )

    fig, axes = plt.subplots(
        2,
        len(SELECTED_COLUMNS),
        figsize=(3.4, 1.8),
        sharey="row",
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.17,
        top=0.80,
        wspace=0.12,
        hspace=0.2,
    )

    for row_idx in range(axes.shape[0]):
        for col_idx in range(2, axes.shape[1]):
            position = axes[row_idx, col_idx].get_position()
            axes[row_idx, col_idx].set_position(
                [
                    position.x0 + COLUMN_GROUP_GAP,
                    position.y0,
                    position.width,
                    position.height,
                ]
            )

    legend_handles = None
    legend_labels = None

    for col_idx, column in enumerate(SELECTED_COLUMNS):
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

        ax_slo.set_title(COLUMN_LABELS[column], pad=3.5)
        ax_slo.grid(True, alpha=0.4)
        ax_acc.grid(True, alpha=0.4)
        ax_slo.set_ylim(*SLO_Y_LIMITS)
        ax_slo.set_yticks(SLO_Y_TICKS)
        ax_acc.set_ylim(*ACC_Y_LIMITS)
        ax_acc.set_yticks(ACC_Y_TICKS)

        ax_slo.tick_params(axis="both", direction="in", length=1.5)
        ax_acc.tick_params(axis="both", direction="in", length=1.5)

        if col_idx == 0:
            ax_slo.set_ylabel("SLO (%)")
            ax_acc.set_ylabel("Acc (%)")
            ax_slo.yaxis.set_label_coords(LEFT_YLABEL_X, 0.5)
            ax_acc.yaxis.set_label_coords(LEFT_YLABEL_X, 0.5)

        x_axis_config = X_AXIS_CONFIG[column]
        if x_axis_config["ticks"] is not None:
            ax_slo.set_xticks(x_axis_config["ticks"])
            ax_acc.set_xticks(x_axis_config["ticks"])
        if x_axis_config["limits"] is not None:
            x_min, x_max = x_axis_config["limits"]
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
            bbox_to_anchor=(0.56, 1.02),
            frameon=True,
            handlelength=1.5,
            columnspacing=1.0,
        )

    second_col_bbox = axes[0, 1].get_position()
    third_col_bbox = axes[0, 2].get_position()
    bottom_row_bbox = axes[1, 0].get_position()
    separator_x = 0.5 * (second_col_bbox.x1 + third_col_bbox.x0)
    fig.add_artist(
        Line2D(
            [separator_x, separator_x],
            [bottom_row_bbox.y0-0.06, second_col_bbox.y1+0.06],
            transform=fig.transFigure,
            color="0.45",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
    )

    fig.supxlabel("RPS per GPU", x=0.55, y=0.02, fontsize=8)

    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "input_output_ablation.txt",
        help="Path to the ablation summary text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "io_ablation_compact.png",
        help="Path to the output plot image.",
    )
    args = parser.parse_args()

    data = parse_summary(args.input)
    plot_summary(data, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()