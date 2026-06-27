import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CACHE_DIR = Path(".plot_cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEMS = ["TeDiServe", "Llumnix", "INFaaS"]
COLORS = {
    "INFaaS": "#134686",
    "Llumnix": "#B89000",
    "TeDiServe": "#FF4F0F",
}
MARKERS = {
    "TeDiServe": "o",
    "Llumnix": "s",
    "INFaaS": "d",
}

NUM_GPUS = 16
SUMMARY_VERSION = 1

METRIC_SPECS = {
    "slo_attainment": {
        "rps_key": "slo_att_rps",
        "slo_key": "slo_att_slo",
        "ylabel": "SLO Attainment",
        "is_accuracy": False,
    },
    "overall_accuracy": {
        "rps_key": "overall_rps",
        "slo_key": "overall_slo",
        "ylabel": "Accuracy",
        "is_accuracy": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot both accuracy benchmark suites into one 4x6 figure with a summary cache."
    )
    parser.add_argument(
        "--base-dir",
        default="/scratch/10446/tchang85/accuracy_benchmark_results",
        help="Base directory containing extracted experiment runs.",
    )
    parser.add_argument(
        "--repo-root",
        default="/u/tchang85/dllm",
        help="Repository root that contains the SLO analysis script.",
    )
    parser.add_argument(
        "--summary",
        default="merged_accuracy_summary.json",
        help="Path to the suite summary cache JSON.",
    )
    parser.add_argument(
        "--output",
        default="system_compare_merged.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--refresh-summary",
        action="store_true",
        help="Recompute the suite summary even if the summary JSON exists.",
    )
    parser.add_argument(
        "--slo-analysis-script",
        default="analysis/slo_attainment_and_good_accuracy/run.py",
        help="Path relative to repo root for the SLO analysis script.",
    )
    return parser.parse_args()


def file_hash(path: str) -> str:
    match = re.search(r"(\d{8}/\d{6})", path)
    if match:
        return match.group(1).replace("/", "_")

    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def cache_load(key: str):
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as fp:
            return json.load(fp)
    return None


def cache_save(key: str, data) -> None:
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as fp:
        json.dump(data, fp)


def extract_rps(exp_dir: str) -> float:
    sh_path = os.path.join(exp_dir, "run_lmeval.sh")
    if not os.path.exists(sh_path):
        raise FileNotFoundError(sh_path)

    with open(sh_path, "r") as f:
        content = f.read()

    match = re.search(r'^\s*ARRIVAL_PATTERN\s*=\s*"([0-9]+):([0-9.]+)', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse ARRIVAL_PATTERN in {exp_dir}")

    interval = float(match.group(2))
    return 1.0 / interval


def extract_rps_cached(exp_dir: str) -> float:
    key = f"rps_{exp_dir.replace('/', '_')}"
    cached = cache_load(key)
    if cached:
        return cached["rps"]

    rps = extract_rps(exp_dir)
    cache_save(key, {"rps": rps})
    return rps


def find_result_json(exp_dir: str) -> str:
    results_dir = os.path.join(exp_dir, "results")
    for root, _, files in os.walk(results_dir):
        for file_name in sorted(files):
            if file_name.endswith(".json"):
                return os.path.join(root, file_name)
    raise FileNotFoundError(f"No JSON in {exp_dir}/results/")


def run_slo_analysis(script_path: str, json_file: str, slo: float):
    cmd = [
        "python",
        script_path,
        "--path",
        json_file,
        "--slo",
        str(slo),
    ]
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    match_slo = re.search(r"SLO attainment.*?: ([0-9.]+)", output)
    match_good = re.search(r"Accuracy.*?: ([0-9.]+)", output)
    match_overall = re.search(r"Overall accuracy.*?: ([0-9.]+)", output)

    slo_att = float(match_slo.group(1)) if match_slo else None
    good = float(match_good.group(1)) if match_good else None
    overall = float(match_overall.group(1)) if match_overall else None
    return slo_att, good, overall


def run_slo_analysis_cached(script_path: str, json_file: str, slo: float):
    key = f"slo_{file_hash(json_file)}_{slo}".replace(".", "_")
    cached = cache_load(key)
    if cached:
        return cached["att"], cached["good"], cached["overall"]

    att, good, overall = run_slo_analysis(script_path, json_file, slo)
    cache_save(key, {"att": att, "good": good, "overall": overall})
    return att, good, overall


def compute_curves_for_benchmark(
    base_dir: str,
    exp_dir_name: str,
    slo_analysis_script: str,
    rps_dirs: dict,
    slo_dirs: dict,
    slo_values: list[float],
    base_slo: float,
) -> dict:
    rps_vals = {system: [] for system in SYSTEMS}
    slo_att_rps = {system: [] for system in SYSTEMS}
    good_rps = {system: [] for system in SYSTEMS}
    overall_rps = {system: [] for system in SYSTEMS}
    slo_att_slo = {system: [] for system in SYSTEMS}
    good_slo = {system: [] for system in SYSTEMS}
    overall_slo = {system: [] for system in SYSTEMS}

    for system in SYSTEMS:
        if system not in rps_dirs:
            continue

        for rel_dir in rps_dirs[system]:
            exp_path = os.path.join(base_dir, exp_dir_name, rel_dir)
            json_file = find_result_json(exp_path)
            rps = extract_rps_cached(exp_path)
            rps_vals[system].append(rps / NUM_GPUS)

            with open(os.path.join(exp_path, "run_lmeval.sh")) as f:
                match = re.search(r"SLO\s*=\s*([0-9.]+)", f.read())
            exp_slo = float(match.group(1))

            att, good, overall = run_slo_analysis_cached(slo_analysis_script, json_file, exp_slo)
            slo_att_rps[system].append(att)
            good_rps[system].append(good)
            overall_rps[system].append(overall)

    slo_sorted = sorted(slo_values, reverse=True)
    slo_multipliers = [value / base_slo for value in slo_sorted]

    for system in SYSTEMS:
        if system not in slo_dirs:
            continue

        dirs = slo_dirs[system]
        if len(dirs) == len(slo_sorted):
            for rel_dir, slo in zip(dirs, slo_sorted):
                exp_path = os.path.join(base_dir, exp_dir_name, rel_dir)
                json_file = find_result_json(exp_path)
                att, good, overall = run_slo_analysis_cached(slo_analysis_script, json_file, slo)
                slo_att_slo[system].append(att)
                good_slo[system].append(good)
                overall_slo[system].append(overall)
        else:
            exp_path = os.path.join(base_dir, exp_dir_name, dirs[0])
            json_file = find_result_json(exp_path)
            for slo in slo_sorted:
                att, good, overall = run_slo_analysis_cached(slo_analysis_script, json_file, slo)
                slo_att_slo[system].append(att)
                good_slo[system].append(good)
                overall_slo[system].append(overall)

    return {
        "rps_vals": rps_vals,
        "slo_multipliers": slo_multipliers,
        "slo_att_rps": slo_att_rps,
        "good_rps": good_rps,
        "overall_rps": overall_rps,
        "slo_att_slo": slo_att_slo,
        "good_slo": good_slo,
        "overall_slo": overall_slo,
    }


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    with open(summary_path, "r") as fp:
        return json.load(fp)


def save_summary(summary_path: Path, payload: dict) -> None:
    with open(summary_path, "w") as fp:
        json.dump(payload, fp, indent=2)


def build_summary(
    base_dir: str,
    exp_dir_name: str,
    slo_analysis_script: str,
    suites: dict,
    summary_path: Path,
    refresh_summary: bool,
) -> dict:
    summary = None if refresh_summary else load_summary(summary_path)
    if summary is None or summary.get("version") != SUMMARY_VERSION:
        summary = {
            "version": SUMMARY_VERSION,
            "base_dir": base_dir,
            "exp_dir_name": exp_dir_name,
            "slo_analysis_script": slo_analysis_script,
            "suites": {},
        }

    for suite_key, suite_cfg in suites.items():
        suite_summary = summary["suites"].setdefault(suite_key, {})
        suite_summary.setdefault("label", suite_cfg["label"])
        suite_summary.setdefault("accuracy_ylims", suite_cfg["accuracy_ylims"])
        suite_summary.setdefault("benchmarks", {})

        for bench_name, bench_cfg in suite_cfg["benchmarks"].items():
            if bench_name in suite_summary["benchmarks"] and not refresh_summary:
                print(f"[summary] Using cached summary for {suite_key}/{bench_name}")
                continue

            print(f"[summary] Computing {suite_key}/{bench_name}")
            suite_summary["benchmarks"][bench_name] = compute_curves_for_benchmark(
                base_dir=base_dir,
                exp_dir_name=exp_dir_name,
                slo_analysis_script=slo_analysis_script,
                rps_dirs=bench_cfg["rps_dirs"],
                slo_dirs=bench_cfg["slo_dirs"],
                slo_values=bench_cfg["slo_values"],
                base_slo=bench_cfg["base_slo"],
            )
            save_summary(summary_path, summary)

    save_summary(summary_path, summary)
    return summary


def plot_merged_suites(summary: dict, save_path: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 20,
            "lines.linewidth": 2.5,
            "lines.markersize": 9,
        }
    )

    suite_order = ["standard", "dream"]
    metrics = ["slo_attainment", "overall_accuracy"]
    bench_order = ["GSM8K", "MMLU-Pro", "MBPP"]
    row_specs = [(suite_key, metric) for suite_key in suite_order for metric in metrics]

    widths = []
    for _ in bench_order:
        widths.extend([0.95, 0.95, 0.1])
    widths = widths[:-1]

    fig = plt.figure(figsize=(27, 13))
    gs = fig.add_gridspec(4, len(widths), width_ratios=widths, wspace=0.15, hspace=0.22)
    axs = np.empty((4, 6), dtype=object)

    for row in range(4):
        for bench_idx in range(len(bench_order)):
            rps_col = 3 * bench_idx
            slo_col = rps_col + 1
            axs[row, 2 * bench_idx] = fig.add_subplot(gs[row, rps_col])
            axs[row, 2 * bench_idx + 1] = fig.add_subplot(gs[row, slo_col])

    legend_handles = None
    legend_labels = None

    for row_idx, (suite_key, metric) in enumerate(row_specs):
        suite_summary = summary["suites"][suite_key]
        accuracy_ylims = suite_summary["accuracy_ylims"]
        metric_spec = METRIC_SPECS[metric]

        for bench_idx, bench_name in enumerate(bench_order):
            curves = suite_summary["benchmarks"][bench_name]
            rps_vals = curves["rps_vals"]
            slo_multipliers = curves["slo_multipliers"]
            y_rps = curves[metric_spec["rps_key"]]
            y_slo = curves[metric_spec["slo_key"]]
            ax_rps = axs[row_idx, 2 * bench_idx]
            ax_slo = axs[row_idx, 2 * bench_idx + 1]

            for system in SYSTEMS:
                xs = [x for x, y in zip(rps_vals[system], y_rps[system]) if y is not None]
                ys = [y for y in y_rps[system] if y is not None]
                if xs:
                    idx = np.argsort(xs)
                    xs_sorted = np.array(xs)[idx]
                    ys_sorted = np.array(ys)[idx]
                    ax_rps.plot(
                        xs_sorted,
                        ys_sorted,
                        marker=MARKERS[system],
                        color=COLORS[system],
                        label=system,
                    )

                slo_ys = [y for y in y_slo[system] if y is not None]
                if slo_ys:
                    ax_slo.plot(
                        slo_multipliers,
                        slo_ys,
                        marker=MARKERS[system],
                        color=COLORS[system],
                    )

            ax_rps.grid(True, alpha=0.4)
            ax_slo.grid(True, alpha=0.4)

            if bench_idx == 0:
                ax_rps.set_ylabel(metric_spec["ylabel"])

            if row_idx == len(row_specs) - 1:
                ax_rps.set_xlabel("RPS per GPU")
                ax_slo.set_xlabel("SLO Multiple")
            else:
                ax_rps.set_xticklabels([])
                ax_slo.set_xticklabels([])

            ax_slo.set_yticklabels([])

            if legend_handles is None:
                legend_handles, legend_labels = ax_rps.get_legend_handles_labels()

            if not metric_spec["is_accuracy"]:
                yticks = np.arange(0, 1.01, 0.25)
                ax_rps.set_ylim(0, 1.05)
                ax_slo.set_ylim(0, 1.05)
                ax_rps.set_yticks(yticks)
                ax_slo.set_yticks(yticks)
            else:
                lo, hi = accuracy_ylims[bench_name]
                yticks = np.arange(np.ceil(lo * 100) / 100, np.floor(hi * 100) / 100 + 0.001, 0.01)
                ax_rps.set_ylim(lo, hi)
                ax_slo.set_ylim(lo, hi)
                ax_rps.set_yticks(yticks)
                ax_slo.set_yticks(yticks)

            if slo_multipliers:
                slo_min = min(slo_multipliers)
                slo_max = max(slo_multipliers)
                lo_int = int(np.ceil(slo_min))
                hi_int = int(np.ceil(slo_max))
                int_ticks = list(range(hi_int, lo_int - 1, -1))
                ax_slo.set_xlim(slo_max, slo_min)
                if row_idx == len(row_specs) - 1:
                    ax_slo.set_xticks(int_ticks)
                    ax_slo.set_xticklabels([f"{tick}x" for tick in int_ticks])

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.92])
    fig.canvas.draw()

    for bench_idx, bench_name in enumerate(bench_order):
        ax_left = axs[0, 2 * bench_idx]
        ax_right = axs[0, 2 * bench_idx + 1]
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        x_center = (pos_l.x0 + pos_r.x1) / 2.0
        y_top = max(pos_l.y1, pos_r.y1)
        fig.text(x_center, y_top + 0.02, bench_name, ha="center", va="bottom", fontsize=22, fontweight="bold")

    suite_labels = {
        "standard": "Standard Suite",
        "dream": "Dream Suite",
    }
    for suite_idx, suite_key in enumerate(suite_order):
        top_ax = axs[2 * suite_idx, 0]
        bottom_ax = axs[2 * suite_idx + 1, 0]
        pos_top = top_ax.get_position()
        pos_bottom = bottom_ax.get_position()
        y_center = (pos_top.y1 + pos_bottom.y0) / 2.0
        fig.text(0.015, y_center, suite_labels[suite_key], rotation=90, va="center", ha="center", fontsize=22, fontweight="bold")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(SYSTEMS),
            bbox_to_anchor=(0.5, 1.01),
        )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved plot: {save_path}")


def get_suites() -> dict:
    return {
        "standard": {
            "label": "Standard Suite",
            "accuracy_ylims": {
                "GSM8K": [0.75, 0.79],
                "MMLU-Pro": [0.33, 0.37],
                "MBPP": [0.30, 0.34],
            },
            "benchmarks": {
                "GSM8K": {
                    "rps_dirs": {
                        "TeDiServe": [
                            "20251127/182647",
                            "20251127/183039",
                            "20251127/185122",
                            "20251208/075520",
                            "20251208/080114",
                        ],
                        "Llumnix": [
                            "20251127/181908",
                            "20251127/183426",
                            "20251127/184703",
                            "20251127/181510",
                            "20251208/080918",
                        ],
                        "INFaaS": [
                            "20251127/182256",
                            "20251127/183818",
                            "20251127/184316",
                            "20251127/142544",
                            "20251208/080515",
                        ],
                    },
                    "slo_dirs": {
                        "TeDiServe": [
                            "20251127/182647",
                            "20251206/212506",
                            "20251206/213519",
                            "20251206/215415",
                            "20251206/214456",
                        ],
                        "Llumnix": ["20251127/181908"],
                        "INFaaS": ["20251127/182256"],
                    },
                    "slo_values": [9.5, 7.6, 5.7, 4.75, 3.8],
                    "base_slo": 1.9,
                },
                "MMLU-Pro": {
                    "rps_dirs": {
                        "TeDiServe": [
                            "20251124/212006",
                            "20251124/204610",
                            "20251125/121423",
                            "20251125/111533",
                            "20251125/124134",
                        ],
                        "Llumnix": [
                            "20251124/185130",
                            "20251125/132058",
                            "20251125/122827",
                            "20251125/112844",
                            "20251125/130719",
                        ],
                        "INFaaS": [
                            "20251124/183458",
                            "20251124/210259",
                            "20251125/103451",
                            "20251125/104806",
                            "20251125/125359",
                        ],
                    },
                    "slo_dirs": {
                        "TeDiServe": [
                            "20251124/204610",
                            "20251125/135207",
                            "20251125/140855",
                            "20251125/142511",
                            "20251125/143948",
                        ],
                        "Llumnix": ["20251125/132058"],
                        "INFaaS": ["20251124/210259"],
                    },
                    "slo_values": [12, 9.6, 8.4, 7.2, 6],
                    "base_slo": 2.4,
                },
                "MBPP": {
                    "rps_dirs": {
                        "TeDiServe": [
                            "20251208/085456",
                            "20251127/164621",
                            "20251127/155756",
                            "20251208/094245",
                            "20251208/094757",
                        ],
                        "Llumnix": [
                            "20251127/163425",
                            "20251127/163859",
                            "20251127/165244",
                            "20251127/165721",
                            "20251127/170216",
                        ],
                        "INFaaS": [
                            "20251127/162553",
                            "20251127/161647",
                            "20251127/151437",
                            "20251127/160716",
                            "20251127/152353",
                        ],
                    },
                    "slo_dirs": {
                        "TeDiServe": [
                            "20251208/085456",
                            "20251208/090914",
                            "20251208/091328",
                            "20251208/093543",
                        ],
                        "Llumnix": ["20251127/163425"],
                        "INFaaS": ["20251127/162553"],
                    },
                    "slo_values": [7.41, 5.928, 4.446, 2.964],
                    "base_slo": 1.482,
                },
            },
        },
        "dream": {
            "label": "Dream Suite",
            "accuracy_ylims": {
                "GSM8K": [0.68, 0.73],
                "MMLU-Pro": [0.325, 0.365],
                "MBPP": [0.345, 0.385],
            },
            "benchmarks": {
                "GSM8K": {
                    "rps_dirs": {
                        "TeDiServe": [
                            "20251208/234925",
                            "20251208/234518",
                            "20251208/225012",
                            "20251208/232220",
                            "20251208/230124",
                        ],
                        "Llumnix": [
                            "20251209/002920",
                            "20251209/000011",
                            "20251209/000406",
                            "20251209/000807",
                            "20251209/001149",
                        ],
                        "INFaaS": [
                            "20251208/223558",
                            "20251208/234122",
                            "20251208/225357",
                            "20251208/231657",
                            "20251208/231227",
                        ],
                    },
                    "slo_dirs": {
                        "TeDiServe": [
                            "20251208/234925",
                            "20251209/001642",
                            "20251209/002051",
                            "20251209/002445",
                        ],
                        "Llumnix": ["20251209/002920"],
                        "INFaaS": ["20251208/223558"],
                    },
                    "slo_values": [8, 6.4, 4.8, 3.2],
                    "base_slo": 1.6,
                },
                "MMLU-Pro": {
                    "rps_dirs": {
                        "TeDiServe": [
                            "20251209/074955",
                            "20251209/073917",
                            "20251209/072844",
                            "20251209/062429",
                        ],
                        "Llumnix": [
                            "20251209/080018",
                            "20251209/081127",
                            "20251209/070333",
                            "20251209/064954",
                        ],
                        "INFaaS": [
                            "20251209/055310",
                            "20251209/094253",
                            "20251209/071704",
                            "20251209/061150",
                        ],
                    },
                    "slo_dirs": {
                        "TeDiServe": [
                            "20251209/074955",
                            "20251209/084454",
                            "20251209/085504",
                            "20251209/090728",
                        ],
                        "Llumnix": ["20251209/080018"],
                        "INFaaS": ["20251209/055310"],
                    },
                    "slo_values": [11.5, 9.2, 6.9, 4.6],
                    "base_slo": 2.3,
                },
                "MBPP": {
                    "rps_dirs": {
                        "TeDiServe": [
                            "20251209/035454",
                            "20251209/032536",
                            "20251209/034702",
                            "20251209/034142",
                            "20251209/040305",
                        ],
                        "Llumnix": [
                            "20251209/041600",
                            "20251209/041955",
                            "20251209/042453",
                            "20251209/042808",
                            "20251209/041138",
                        ],
                        "INFaaS": [
                            "20251209/043201",
                            "20251209/031538",
                            "20251209/035105",
                            "20251209/031102",
                            "20251209/035918",
                        ],
                    },
                    "slo_dirs": {
                        "TeDiServe": [
                            "20251209/035454",
                            "20251209/043510",
                            "20251209/044050",
                            "20251209/045015",
                        ],
                        "Llumnix": ["20251209/041600"],
                        "INFaaS": ["20251209/043201"],
                    },
                    "slo_values": [10, 8, 6, 5],
                    "base_slo": 2,
                },
            },
        },
    }


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    exp_dir_name = ""
    slo_analysis_script = os.path.join(args.repo_root, args.slo_analysis_script)
    suites = get_suites()
    summary_path = Path(args.summary)

    summary = build_summary(
        base_dir=base_dir,
        exp_dir_name=exp_dir_name,
        slo_analysis_script=slo_analysis_script,
        suites=suites,
        summary_path=summary_path,
        refresh_summary=args.refresh_summary,
    )
    plot_merged_suites(summary, args.output)


if __name__ == "__main__":
    main()