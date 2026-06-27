from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("steps_left_results.txt")
OUTPUT_STEM = "granularity_tradeoff"
FONT_SIZE = 7


def parse_results(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    pattern = re.compile(
        r"Granularity:\s*([0-9.]+),\s*Overhead:\s*([0-9.]+),\s*WMAPE:\s*([0-9.]+)"
    )

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = pattern.fullmatch(line)
            if match is None:
                raise ValueError(f"Unrecognized line format: {line}")

            rows.append(
                {
                    "granularity": float(match.group(1)),
                    "overhead": float(match.group(2)),
                    "mean_request_wmape": float(match.group(3)),
                }
            )

    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    
    # remove one-shot
    rows = [row for row in rows if row["granularity"] != 100000]
    


    return pd.DataFrame(rows)


def plot_granularity_tradeoff(granularity_summary_df: pd.DataFrame, out_dir: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(1.7, 1.23))
    ax.tick_params(axis="both", direction="in", pad=1.0)
    ax.grid(True, alpha=0.28, linestyle="--")

    # y-ticks at 20, 24, 28
    ax.set_yticks([20, 22, 24, 26, 28])

    # make sure x starts from 0
    # ax.set_xlim(0, 20.5)

    df = granularity_summary_df.sort_values("granularity").copy()
    df["overhead_pct"] = df["overhead"] * 100
    df["wmape_pct"] = df["mean_request_wmape"] * 100

    norm = plt.Normalize(
        np.log2(df[df["granularity"] != 100000]["granularity"]).min(),
        np.log2(df[df["granularity"] != 100000]["granularity"]).max(),
    )
    cmap = plt.cm.viridis

    for _, row in df.iterrows():
        granularity = int(row["granularity"])

        if granularity == 100000:
            color = "white"
            label = "One-shot"
        else:
            color = cmap(norm(np.log2(granularity)))
            label = f"{granularity}"

        ax.scatter(
            row["overhead_pct"],
            row["wmape_pct"],
            color=color,
            edgecolors="black",
            s=38,
            linewidths=0.6,
            label=label,
            zorder=3,
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_labels = ["1", "16", "2", "32", "4", "64", "8"]
    legend_handles = [by_label[label] for label in legend_labels if label in by_label]

    ax.set_xlabel("Overhead (%)", labelpad=1)
    ax.set_ylabel("Pred. Error (%)", labelpad=1)

    baseline_wmape = (
        df[df["granularity"] == 1]["wmape_pct"].values[0]
        if 1 in df["granularity"].values
        else df[df["granularity"] == df["granularity"].min()]["wmape_pct"].values[0]
    )
    baseline_wmape *= 1.05
    # make sure line on top of dots
    ax.axhline(y=baseline_wmape, color="red", linestyle="--", linewidth=1.5, zorder=2)

    legend = ax.legend(
        legend_handles,
        legend_labels,
        title="Granularity\n(tokens unmasked)",
        fontsize=FONT_SIZE - 1,
        title_fontsize=FONT_SIZE,
        loc="best",
        ncol=4,
        borderpad=0.35,
        handlelength=0.8,
        handletextpad=0.45,
        labelspacing=0.4,
        columnspacing=0.4,
    )
    legend.get_title().set_ha("center")
    legend._legend_box.align = "center"

    fig.tight_layout(pad=0.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{OUTPUT_STEM}.png"
    pdf_path = out_dir / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


if __name__ == "__main__":
    plot_granularity_tradeoff(parse_results(INPUT_FILE), Path("."))