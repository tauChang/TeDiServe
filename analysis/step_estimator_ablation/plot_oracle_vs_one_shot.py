#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROW_RE = re.compile(r"^(oracle|lightgbm)_slo_(\d+)_rps_(\d+)$")


def _to_float_or_none(value: str):
    value = value.strip()
    if value in {"N/A", "ERROR", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_summary_rows(summary_path: Path):
    rows = []
    for raw_line in summary_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        if line.startswith("experiment") or line.startswith("-"):
            continue
        if line.startswith("===") or line.startswith("experiments:"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 9:
            continue

        name = parts[0]
        m = ROW_RE.match(name)
        if not m:
            continue

        method_raw, slo_str, rps_str = m.groups()
        method = "one-shot" if method_raw == "lightgbm" else "oracle"
        row = {
            "name": name,
            "method": method,
            "slo": int(slo_str),
            "rps": int(rps_str),
            "slo_attainment": _to_float_or_none(parts[1]),
            "accuracy": _to_float_or_none(parts[2]),
            "avg_confidence": _to_float_or_none(parts[8]),
        }
        rows.append(row)

    return rows


def plot_metrics(rows, output_path: Path):
    metrics = [
        ("slo_attainment", "SLO Attainment", "Ratio"),
        ("accuracy", "Accuracy", "Ratio"),
        ("avg_confidence", "Average Confidence", "Ratio"),
    ]

    slos = sorted({r["slo"] for r in rows})
    methods = ["oracle", "one-shot"]

    if not slos:
        raise ValueError("No valid rows found in summary.")

    colors = {s: c for s, c in zip(slos, ["tab:blue", "tab:orange", "tab:green", "tab:red"]) }
    linestyles = {"oracle": "-", "one-shot": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        for slo in slos:
            for method in methods:
                filtered = [
                    r for r in rows
                    if r["slo"] == slo and r["method"] == method and r[metric_key] is not None
                ]
                filtered.sort(key=lambda x: x["rps"])
                if not filtered:
                    continue

                x = [r["rps"] for r in filtered]
                y = [r[metric_key] for r in filtered]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    linestyle=linestyles[method],
                    color=colors[slo],
                    label=f"SLO={slo} {method}",
                )

        ax.set_title(title)
        ax.set_xlabel("RPS")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize oracle vs one-shot metrics from final_summary.txt"
    )
    parser.add_argument("--summary-path", required=True, help="Path to final_summary.txt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (default: same directory as summary file)",
    )
    parser.add_argument(
        "--output-name",
        default="oracle_vs_one_shot_metrics.png",
        help="Output image filename",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    rows = load_summary_rows(summary_path)
    if not rows:
        raise RuntimeError("No parseable oracle/lightgbm rows found in summary file.")

    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    output_path = output_dir / args.output_name

    plot_metrics(rows, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
