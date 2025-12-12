import os
import re
import json
import subprocess
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#                  GLOBAL CONSTANTS
# ============================================================

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

NUM_GPUS = 16  # for per-GPU RPS

# Per-benchmark accuracy y-limits (3 percentage points each)
ACCURACY_YLIMS = {
    "GSM8K": (0.77, 0.79),
    "MBPP": (0.39, 0.43),
}


# ============================================================
#                     CACHE HELPERS
# ============================================================

def file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cache_load(key: str):
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        return json.load(open(f, "r"))
    return None


def cache_save(key: str, data):
    json.dump(data, open(CACHE_DIR / f"{key}.json", "w"))


# ============================================================
#                     PARSING HELPERS
# ============================================================

def extract_rps(exp_dir: str) -> float:
    """Parse ARRIVAL_PATTERN from run_lmeval.sh as 1/interarrival."""
    sh_path = os.path.join(exp_dir, "run_lmeval.sh")
    if not os.path.exists(sh_path):
        raise FileNotFoundError(sh_path)

    with open(sh_path, "r") as f:
        c = f.read()

    m = re.search(r'^\s*ARRIVAL_PATTERN\s*=\s*"([0-9]+):([0-9.]+)', c, re.MULTILINE)
    if not m:
        raise ValueError(f"Cannot parse ARRIVAL_PATTERN in {exp_dir}")

    interval = float(m.group(2))
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
        for f in files:
            if f.endswith(".json"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No result JSON under {exp_dir}/results/")


# ============================================================
#         COMPUTE ONLY RPS-SWEEP CURVES FOR ONE BENCHMARK
# ============================================================

def compute_rps_curves(
    base_dir, exp_dir_name,
    rps_dirs,
):
    rps_vals = {sys: [] for sys in SYSTEMS}
    slo_att_rps = {sys: [] for sys in SYSTEMS}
    overall_rps = {sys: [] for sys in SYSTEMS}
    good_rps = {sys: [] for sys in SYSTEMS}

    for sys in SYSTEMS:
        if sys not in rps_dirs:
            continue

        for d in rps_dirs[sys]:
            exp_path = os.path.join(base_dir, exp_dir_name, d)
            json_file = find_result_json(exp_path)

            # per-GPU RPS
            rps = extract_rps_cached(exp_path) / NUM_GPUS
            rps_vals[sys].append(rps)

            # run SLO analysis using *experiment SLO*
            with open(os.path.join(exp_path, "run_lmeval.sh")) as f:
                m = re.search(r"SLO\s*=\s*([0-9.]+)", f.read())
            exp_slo = float(m.group(1))

            # ----- Cache key for SLO analysis -----
            json_hash = file_hash(json_file)
            cache_key = f"slo_{json_hash}_slo{exp_slo}".replace(".", "_")

            cached = cache_load(cache_key)
            if cached:
                slo_val = cached["slo"]
                good_val = cached["good"]
                overall_val = cached["overall"]
            else:
                # Run external script only if not cached
                cmd = [
                    "python", "/work2/10446/tchang85/stampede3/dllm/analysis/slo_attainment_and_good_accuracy/run.py",
                    "--path", json_file,
                    "--slo", str(exp_slo),
                ]
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

                m_s = re.search(r"SLO attainment.*?: ([0-9.]+)", out)
                m_g = re.search(r"Accuracy.*?: ([0-9.]+)", out)
                m_o = re.search(r"Overall accuracy.*?: ([0-9.]+)", out)

                slo_val = float(m_s.group(1))
                good_val = float(m_g.group(1))
                overall_val = float(m_o.group(1))

                cache_save(cache_key, {
                    "slo": slo_val,
                    "good": good_val,
                    "overall": overall_val,
                })

            # Append cached or fresh values
            slo_att_rps[sys].append(slo_val)
            good_rps[sys].append(good_val)
            overall_rps[sys].append(overall_val)


    return {
        "rps_vals": rps_vals,
        "slo_att_rps": slo_att_rps,
        "good_rps": good_rps,
        "overall_rps": overall_rps,
    }


# ============================================================
#                   PLOT RPS SWEEP ONLY
# ============================================================

def plot_rps_only(
    base_dir,
    benchmarks,
    metrics_to_plot,
    save_path="rps_only_compare.png",
    exp_dir_name="experiment_dir",
):
    # plotting style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
    })

    bench_names = list(benchmarks.keys())
    num_bench = len(bench_names)
    num_metrics = len(metrics_to_plot)

    fig = plt.figure(figsize=(4.5 * num_bench, 3.0 * num_metrics))
    gs = fig.add_gridspec(
        num_metrics, num_bench,
        wspace=0.15, hspace=0.30,
    )

    axs = np.empty((num_metrics, num_bench), dtype=object)
    for m in range(num_metrics):
        for b in range(num_bench):
            axs[m, b] = fig.add_subplot(gs[m, b])

    legend_handles = None
    legend_labels = None

    # helper
    def plot_rps_curve(ax, xdict, ydict):
        for sys in SYSTEMS:
            xs = xdict[sys]
            ys = ydict[sys]

            xs = [x for x, y in zip(xs, ys) if y is not None]
            ys = [y for y in ys if y is not None]

            if len(xs) == 0:
                continue

            idx = np.argsort(xs)
            xs_sorted = np.array(xs)[idx]
            ys_sorted = np.array(ys)[idx]

            ax.plot(xs_sorted, ys_sorted,
                    marker=MARKERS[sys],
                    color=COLORS[sys],
                    label=sys)

    # main loop
    for b_idx, bench_name in enumerate(bench_names):
        cfg = benchmarks[bench_name]

        curves = compute_rps_curves(
            base_dir, exp_dir_name,
            cfg["rps_dirs"],
        )

        rps_vals = curves["rps_vals"]

        acc_ylim = ACCURACY_YLIMS.get(bench_name, None)

        for m_i, metric in enumerate(metrics_to_plot):
            if metric == "slo_attainment":
                y_rps = curves["slo_att_rps"]
                ylabel = "SLO Attainment"
                is_accuracy = False
            else:
                y_rps = curves["overall_rps"]
                ylabel = "Accuracy"
                is_accuracy = True

            ax = axs[m_i, b_idx]

            plot_rps_curve(ax, rps_vals, y_rps)
            ax.grid(True, alpha=0.4)

            if b_idx == 0:
                ax.set_ylabel(ylabel)

            if m_i == num_metrics - 1:
                ax.set_xlabel("RPS per GPU")
            else:
                ax.set_xticklabels([])

            # y-limits
            if not is_accuracy:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.arange(0, 1.01, 0.25))
            else:
                if acc_ylim is not None:
                    lo, hi = acc_ylim
                    ax.set_ylim(lo, hi)
                    yticks = np.arange(np.ceil(lo * 100) / 100,
                                       np.floor(hi * 100) / 100 + 0.001,
                                       0.01)
                    ax.set_yticks(yticks)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Benchmark title (top-center above both metrics)
        ax_top = axs[0, b_idx]
        pos = ax_top.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.03,
            bench_name,
            ha="center", va="bottom",
            fontsize=22, fontweight="bold"
        )

    # Legend at top
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center",
        ncol=len(SYSTEMS),
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved plot: {save_path}")


# ============================================================
#                       EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    base_dir = "/work2/10446/tchang85/stampede3/dllm"

    benchmarks = {
        "GSM8K": {
            "rps_dirs": {
                "TeDiServe": [
                    "20251202/215415",
                    "20251202/221905",
                    "20251202/212510",
                    "20251202/230216",
                    "20251202/231210",
                ],
                "Llumnix": [
                    "20251202/214552",
                    "20251202/221128",
                    "20251202/211700",
                    "20251202/223515",
                    "20251202/233317",
                ],
                "INFaaS": [
                    "20251202/213722",
                    "20251202/220254",
                    "20251202/210936",
                    "20251202/222804",
                    "20251202/232241",
                ],
            },
        },
        "MBPP": {
            "rps_dirs": {
                "TeDiServe": [
                    "20251203/014657",
                    "20251203/024445",
                    "20251203/010835",
                    "20251203/020641",
                    "20251203/012122",
                ],
                "Llumnix": [
                    "20251203/014007",
                    "20251203/023817",
                    "20251203/013401",
                    "20251203/020007",
                    "20251203/012745",
                ],
                "INFaaS": [
                    "20251203/005142",
                    "20251203/023109",
                    "20251203/005833",
                    "20251203/015409",
                    "20251203/011524",
                ],
            },
        },
    }

    metrics = ["slo_attainment", "overall_accuracy"]

    plot_rps_only(
        base_dir,
        benchmarks,
        metrics_to_plot=metrics,
        save_path="rps_only_compare.png",
    )
