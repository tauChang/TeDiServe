from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_STEM = "step_estimation_features_ablation"

ABLATION_ROWS = [
    ("Output Len", 0.7502, "#2ca02c"),
    ("+ Progress\nStats", 0.3515, "#f28e2b"),
    ("+ Confidence\nStats", 0.1926, "#c43c39"),
    ("Oracle", 0.0, "#1f77b4"),
]


def plot_ablation(out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )

    labels = [row[0] for row in ABLATION_ROWS]
    values = [row[1] for row in ABLATION_ROWS]
    colors = [row[2] for row in ABLATION_ROWS]
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(1.75, 1.35))
    bars = ax.barh(positions, values, color=colors, height=0.8)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pred. Error (%)", labelpad=1)
    ax.tick_params(axis="x", direction="in", pad=2.0)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    max_value = max(values)
    # ax.set_xlim(0, max_value * 1.14)
    ax.set_xlim(0, 0.78)
    # set ticks at 0.25
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["0", "25", "50", "75"])

    for bar, value in zip(bars, values):
        if value < 0.7:
            y_center = bar.get_y() + bar.get_height() / 2
            x_text = value + max_value * 0.01 if value > 0 else max_value * 0.02
            # make text three decimals
            ax.text(x_text, y_center, f"{value * 100:.1f}", va="center", ha="left", fontsize=6.5, fontweight="bold")
        else:
            # y_text = bar.get_y() + bar.get_height() + 0.06
            y_text = bar.get_y() + bar.get_height()/2-0.15
            x_text = value-0.02 if value > 0 else max_value * 0.02
            # make etxt white and bold
            ax.text(x_text, y_text, f"{value * 100:.1f}", va="top", ha="right", fontsize=6.5, color="white", fontweight="bold")

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{OUTPUT_STEM}.png"
    pdf_path = out_dir / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


if __name__ == "__main__":
    plot_ablation(Path("."))