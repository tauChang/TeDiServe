import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


BASE_COLORS = {
    "oracle": "#1f77b4",
    "lightgbm_32": "#d62728",
    "lightgbm_1000": "#2ca02c",
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
                confidence = float(parts[8])
            except ValueError:
                continue

            rows.append(
                {
                    "experiment": exp_name,
                    "slo_attainment": slo_attainment,
                    "confidence": confidence,
                }
            )
    return rows


def model_key(experiment_name: str):
    if experiment_name.startswith("oracle_"):
        return "oracle"
    if "lightgbm_delta_32" in experiment_name:
        return "lightgbm_32"
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


def plot_group(rows, order_values, vary_key, title, out_path: Path):
    grouped = {"oracle": [], "lightgbm_32": [], "lightgbm_1000": []}

    for row in rows:
        grouped[row["model"]].append(row)

    for model in grouped:
        grouped[model].sort(key=lambda x: order_values.index(x[vary_key]))

    fig, ax = plt.subplots(figsize=(8, 6))

    shade_levels = [0.0, 0.2, 0.4, 0.6]

    for model, points in grouped.items():
        if not points:
            continue

        x = [p["confidence"] for p in points]
        y = [p["slo_attainment"] for p in points]

        base_color = BASE_COLORS[model]
        ax.plot(x, y, color=base_color, linewidth=2, label=model)

        for idx, p in enumerate(points):
            marker_color = blend_with_white(base_color, shade_levels[idx])
            ax.scatter(
                p["confidence"],
                p["slo_attainment"],
                color=marker_color,
                edgecolor=base_color,
                s=70,
                zorder=3,
            )
            ax.annotate(
                f"{vary_key}={p[vary_key]}",
                (p["confidence"], p["slo_attainment"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=base_color,
            )

    ax.set_xlabel("confidence")
    ax.set_ylabel("slo attainment")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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
                "layout": layout,
                "model": model,
            }
        )

    plot1_rows = [
        r
        for r in normalized
        if r["layout"] == "slo_first" and r["slo"] == 10 and r["rps"] in [18, 16, 14, 12]
    ]

    plot2_rows = [
        r
        for r in normalized
        if r["layout"] == "rps_first" and r["rps"] == 14 and r["slo"] in [10, 8, 6, 4]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_group(
        plot1_rows,
        order_values=[18, 16, 14, 12],
        vary_key="rps",
        title="SLO=10, varying RPS: confidence vs SLO attainment",
        out_path=output_dir / "conf_vs_slo_slo10_varying_rps.png",
    )

    plot_group(
        plot2_rows,
        order_values=[10, 8, 6, 4],
        vary_key="slo",
        title="RPS=14, varying SLO: confidence vs SLO attainment",
        out_path=output_dir / "conf_vs_slo_rps14_varying_slo.png",
    )


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