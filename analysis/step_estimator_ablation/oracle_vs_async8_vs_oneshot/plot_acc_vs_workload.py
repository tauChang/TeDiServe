import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


FONT_SIZE = 24
LINE_WIDTH = 4
DOT_SIZE = 180


BASE_COLORS = {
    "oracle": "#1f77b4",
    "lightgbm_8": "#d62728",
    "lightgbm_1000": "#2ca02c",
}

MODEL_ZORDER = {
    "oracle": 3,
    "lightgbm_8": 2,
    "lightgbm_1000": 1,
}

LEGEND_LABELS = {
    "oracle": "Oracle",
    "lightgbm_8": "TeDiServe",
    "lightgbm_1000": "One-shot",
}


def parse_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("===") or line.startswith("---"):
                continue
            if "|" not in line:
                continue
            if line.startswith("experiment"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 9:
                continue

            exp_name = parts[0]
            try:
                slo_attainment = float(parts[1])
                accuracy = float(parts[2])
                confidence = float(parts[8])
            except ValueError:
                continue

            rows.append(
                {
                    "experiment": exp_name,
                    "slo_attainment": slo_attainment,
                    "accuracy": accuracy,
                    "confidence": confidence,
                }
            )
    return rows


def model_key(experiment_name: str):
    if experiment_name.startswith("oracle_"):
        return "oracle"
    if "lightgbm_delta_8" in experiment_name:
        return "lightgbm_8"
    if "lightgbm_delta_1000" in experiment_name:
        return "lightgbm_1000"
    return None


def parse_slo_rps(experiment_name: str):
    m = re.search(r"slo_(\d+)_rps_(\d+)", experiment_name)
    if m:
        return int(m.group(1)), int(m.group(2)), "slo_first"

    m = re.search(r"rps_(\d+)_slo_(\d+)", experiment_name)
    if m:
        return int(m.group(2)), int(m.group(1)), "rps_first"

    return None, None, None


def blend_with_white(hex_color: str, alpha: float):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r_new = int((1 - alpha) * r + alpha * 255)
    g_new = int((1 - alpha) * g + alpha * 255)
    b_new = int((1 - alpha) * b + alpha * 255)
    return f"#{r_new:02x}{g_new:02x}{b_new:02x}"


def plot_group_on_axis(
    ax,
    rows,
    order_values,
    vary_key,
    y_key,
    title,
    reverse_x: bool = False,
    tick_labels=None,
):
    grouped = {"oracle": [], "lightgbm_8": [], "lightgbm_1000": []}

    for row in rows:
        grouped[row["model"]].append(row)

    for model in grouped:
        grouped[model].sort(key=lambda x: order_values.index(x[vary_key]))

    shade_levels = [0.0, 0.2, 0.4, 0.6]

    for model, points in grouped.items():
        if not points:
            continue

        x = [p[vary_key] for p in points]
        y = [p[y_key] for p in points]

        base_color = BASE_COLORS[model]
        zorder = MODEL_ZORDER[model]
        ax.plot(x, y, color=base_color, linewidth=LINE_WIDTH, zorder=zorder)

        for idx, p in enumerate(points):
            ax.scatter(
                p[vary_key],
                p[y_key],
                color=base_color,
                edgecolor=base_color,
                s=DOT_SIZE,
                zorder=zorder,
            )

    # if title:
    #     ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xticks(order_values)
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels)
    if reverse_x:
        # Use explicit limits instead of invert_xaxis to avoid double-toggling
        # when this axis is shared across subplots.
        ax.set_xlim(max(order_values), min(order_values))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)


def main(summary_path: Path, output_dir: Path):
    rows = parse_summary(summary_path)

    normalized = []
    for row in rows:
        slo, rps, layout = parse_slo_rps(row["experiment"])
        model = model_key(row["experiment"])
        if slo is None or rps is None or model is None:
            continue

        normalized.append(
            {
                **row,
                "slo": slo,
                "rps": rps,
                "slo_multiple": slo / 2.0,
                "rps_per_gpu": rps / 8.0,
                "layout": layout,
                "model": model,
            }
        )

    plot1_rows = [
        r
        for r in normalized
        if r["layout"] == "slo_first" and r["slo"] == 10 and r["rps"] in [16, 15, 14, 13]
    ]

    plot2_rows = [
        r
        for r in normalized
        if r["layout"] == "rps_first" and r["rps"] == 14 and r["slo"] in [10, 8, 6, 4]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "font.family": "serif",
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex="col")

    rps = [13, 14, 15, 16]
    plot_group_on_axis(
        axes[0, 0],
        plot1_rows,
        # order_values=[12 / 8.0, 14 / 8.0, 16 / 8.0, 18 / 8.0],
        order_values=[r / 8.0 for r in rps],
        vary_key="rps_per_gpu",
        y_key="slo_attainment",
        title="Varying RPS (SLO=10)",
        tick_labels=["{:.2f}".format(r / 8.0) for r in rps],
    )

    plot_group_on_axis(
        axes[0, 1],
        plot2_rows,
        order_values=[2, 3, 4, 5],
        vary_key="slo_multiple",
        y_key="slo_attainment",
        title="Varying SLO (RPS=14)",
        reverse_x=True,
        tick_labels=["2x", "3x", "4x", "5x"],
    )

    plot_group_on_axis(
        axes[1, 0],
        plot1_rows,
        order_values=[r / 8.0 for r in rps],
        vary_key="rps_per_gpu",
        y_key="confidence",
        title="",
        tick_labels=["{:.2f}".format(r / 8.0) for r in rps],
    )

    plot_group_on_axis(
        axes[1, 1],
        plot2_rows,
        order_values=[2, 3, 4, 5],
        vary_key="slo_multiple",
        y_key="confidence",
        title="",
        reverse_x=True,
        tick_labels=["2x", "3x", "4x", "5x"],
    )

    plot_group_on_axis(
        axes[2, 0],
        plot1_rows,
        order_values=[r / 8.0 for r in rps],
        vary_key="rps_per_gpu",
        y_key="accuracy",
        title="",
        tick_labels=["{:.2f}".format(r / 8.0) for r in rps],
    )

    plot_group_on_axis(
        axes[2, 1],
        plot2_rows,
        order_values=[2, 3, 4, 5],
        vary_key="slo_multiple",
        y_key="accuracy",
        title="",
        reverse_x=True,
        tick_labels=["2x", "3x", "4x", "5x"],
    )

    # Shared axis labels per requested layout.
    axes[2, 0].set_xlabel("RPS / GPU", fontsize=FONT_SIZE)
    axes[2, 1].set_xlabel("SLO Multiple", fontsize=FONT_SIZE)
    axes[0, 0].set_ylabel("SLO Attainment", fontsize=FONT_SIZE-4)
    axes[1, 0].set_ylabel("Conf. / Token", fontsize=FONT_SIZE)
    axes[2, 0].set_ylabel("Accuracy", fontsize=FONT_SIZE)

    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 0].set_yticks([0.00, 0.50,1.00])
    axes[0, 1].set_yticks([0.00,  0.50, 1.00])
    
    axes[1, 0].set_ylim(0.5, 0.92)
    axes[1, 1].set_ylim(0.5, 0.92)
    # axes[1, 0].set_yticks([0.70, 0.80, 0.90])
    # axes[1, 1].set_yticks([0.70, 0.80, 0.90])

    axes[2, 0].set_ylim(0.65, 0.80)
    axes[2, 1].set_ylim(0.65, 0.80)
    axes[2, 0].set_yticks([0.65, 0.70, 0.75, 0.80])
    axes[2, 1].set_yticks([0.65, 0.70, 0.75, 0.80])

    # Hide redundant labels: x labels only on bottom row, y labels only on left column.
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.tick_params(labelbottom=False)
    for ax in [axes[0, 1], axes[1, 1], axes[2, 1]]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BASE_COLORS[k],
            lw=LINE_WIDTH,
            marker="o",
            markersize=12,
            label=LEGEND_LABELS[k],
        )
        for k in ["oracle", "lightgbm_8", "lightgbm_1000"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=FONT_SIZE,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.subplots_adjust(wspace=0.13, hspace=0.12)
    fig.savefig(output_dir / "acc_conf_vs_workload_2x2.png", dpi=220)
    fig.savefig(output_dir / "acc_conf_vs_workload_2x2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to final_summary.txt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output figures",
    )
    args = parser.parse_args()
    main(args.summary, args.out_dir)