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
    "GSM8K": (0.75, 0.79),
    "MMLU-Pro": (0.33, 0.37),
    "MBPP": (0.30, 0.34),
}

# ============================================================
#                  CACHE UTILITIES
# ============================================================

def file_hash(path: str) -> str:
    """Stable SHA1 hash of file contents."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cache_load(key: str):
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        with open(f, "r") as fp:
            return json.load(fp)
    return None


def cache_save(key: str, data):
    f = CACHE_DIR / f"{key}.json"
    with open(f, "w") as fp:
        json.dump(data, fp)


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

    m = re.search(
        r'^\s*ARRIVAL_PATTERN\s*=\s*"([0-9]+):([0-9.]+)',
        c, re.MULTILINE,
    )
    if not m:
        raise ValueError(f"Could not parse ARRIVAL_PATTERN in {exp_dir}")

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
    raise FileNotFoundError(f"No JSON in {exp_dir}/results/")


# ============================================================
#                  SLO ANALYSIS CACHED WRAPPER
# ============================================================

def run_slo_analysis(script_path: str, json_file: str, slo: float):
    cmd = [
        "python", script_path,
        "--path", json_file,
        "--slo", str(slo),
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    m_slo = re.search(r"SLO attainment.*?: ([0-9.]+)", out)
    m_good = re.search(r"Accuracy.*?: ([0-9.]+)", out)
    m_over = re.search(r"Overall accuracy.*?: ([0-9.]+)", out)

    slo_att = float(m_slo.group(1)) if m_slo else None
    good = float(m_good.group(1)) if m_good else None
    overall = float(m_over.group(1)) if m_over else None

    return slo_att, good, overall


def run_slo_analysis_cached(script_path: str, json_file: str, slo: float):
    h = file_hash(json_file)
    key = f"slo_{h}_{slo}".replace(".", "_")

    c = cache_load(key)
    if c:
        return c["att"], c["good"], c["overall"]

    att, good, overall = run_slo_analysis(script_path, json_file, slo)
    cache_save(key, {"att": att, "good": good, "overall": overall})
    return att, good, overall


# ============================================================
#                     GLOBAL METRIC SPECS
# ============================================================

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


# ============================================================
#        COMPUTE CURVES FOR A SINGLE BENCHMARK
# ============================================================

def compute_curves_for_benchmark(
    base_dir, exp_dir_name,
    slo_analysis_script,
    rps_dirs, slo_dirs, slo_values, base_slo,
):
    rps_vals = {sys: [] for sys in SYSTEMS}

    slo_att_rps = {sys: [] for sys in SYSTEMS}
    good_rps = {sys: [] for sys in SYSTEMS}
    overall_rps = {sys: [] for sys in SYSTEMS}

    slo_att_slo = {sys: [] for sys in SYSTEMS}
    good_slo = {sys: [] for sys in SYSTEMS}
    overall_slo = {sys: [] for sys in SYSTEMS}

    # ----- RPS sweep -----
    for sys in SYSTEMS:
        if sys not in rps_dirs:
            continue

        for d in rps_dirs[sys]:
            exp_path = os.path.join(base_dir, exp_dir_name, d)
            json_file = find_result_json(exp_path)
            rps = extract_rps_cached(exp_path)

            # convert to per-GPU RPS
            rps_vals[sys].append(rps / NUM_GPUS)

            # read experiment SLO from script
            with open(os.path.join(exp_path, "run_lmeval.sh")) as f:
                m = re.search(r"SLO\s*=\s*([0-9.]+)", f.read())
            exp_slo = float(m.group(1))

            att, g, o = run_slo_analysis_cached(
                os.path.join(base_dir, slo_analysis_script),
                json_file, exp_slo,
            )
            slo_att_rps[sys].append(att)
            good_rps[sys].append(g)
            overall_rps[sys].append(o)

    # ----- SLO sweep -----
    slo_sorted = sorted(slo_values, reverse=True)
    slo_multipliers = [s / base_slo for s in slo_sorted]

    for sys in SYSTEMS:
        if sys not in slo_dirs:
            continue

        dirs = slo_dirs[sys]

        # Case 1: one directory per SLO
        if len(dirs) == len(slo_sorted):
            for d, slo in zip(dirs, slo_sorted):
                exp_path = os.path.join(base_dir, exp_dir_name, d)
                json_file = find_result_json(exp_path)

                att, g, o = run_slo_analysis_cached(
                    os.path.join(base_dir, slo_analysis_script),
                    json_file, slo,
                )
                slo_att_slo[sys].append(att)
                good_slo[sys].append(g)
                overall_slo[sys].append(o)

        # Case 2: one directory, sweep SLOs on same JSON
        else:
            exp_path = os.path.join(base_dir, exp_dir_name, dirs[0])
            json_file = find_result_json(exp_path)

            for slo in slo_sorted:
                att, g, o = run_slo_analysis_cached(
                    os.path.join(base_dir, slo_analysis_script),
                    json_file, slo,
                )
                slo_att_slo[sys].append(att)
                good_slo[sys].append(g)
                overall_slo[sys].append(o)

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


# ============================================================
#                MAIN MULTI-BENCHMARK PLOTTER
# ============================================================

def plot_multi_benchmarks(
    base_dir, benchmarks,
    metrics_to_plot,
    save_path="system_compare_multi.png",
    exp_dir_name="experiment_dir",
    slo_analysis_script="analysis/slo_attainment_and_good_accuracy/run.py",
):
    # Validate metrics
    for m in metrics_to_plot:
        if m not in METRIC_SPECS:
            raise ValueError(f"Invalid metric '{m}'")

    # Global style (Droid Serif if available)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "lines.linewidth": 2.5,
        "lines.markersize": 10
    })
    num_bench = len(benchmarks)
    num_metrics = len(metrics_to_plot)
    widths = []

    for _ in range(num_bench):
        widths.extend([.95, .95, 0.1])   # RPS, SLO, GAP

    widths = widths[:-1]   # remove trailing gap

    fig = plt.figure(figsize=(4.5 * 2 * num_bench, 3.0 * num_metrics))
    gs = fig.add_gridspec(
        num_metrics,
        len(widths),
        width_ratios=widths,
        wspace=0.15,
        hspace=0.20,
    )

    axs = np.empty((num_metrics, 2 * num_bench), dtype=object)

    for m in range(num_metrics):
        for b in range(num_bench):
            rps_col = 3 * b      # e.g., 0,3,6,...
            slo_col = rps_col+1  # e.g., 1,4,7,...

            axs[m, 2*b]   = fig.add_subplot(gs[m, rps_col])
            axs[m, 2*b+1] = fig.add_subplot(gs[m, slo_col])



    # num_bench = len(benchmarks)
    # num_metrics = len(metrics_to_plot)

    # # Figure & axes with tighter spacing and narrower width
    # fig, axs = plt.subplots(
    #     num_metrics,
    #     2 * num_bench,
    #     figsize=(4 * 2 * num_bench, 3.0 * num_metrics),
    #     gridspec_kw={"wspace": 0.2, "hspace": 0.20},
    # )

    if num_metrics == 1:
        axs = axs.reshape(1, -1)

    legend_handles = None
    legend_labels = None

    # --------------------------------------------------------
    #         HELPER PLOTTERS
    # --------------------------------------------------------

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

            ax.plot(
                xs_sorted,
                ys_sorted,
                marker=MARKERS[sys],
                color=COLORS[sys],
                label=sys,
            )

    def plot_slo_curve(ax, slo_mult, ydict):
        for sys in SYSTEMS:
            ys = ydict[sys]
            ys = [y for y in ys if y is not None]
            if len(ys) == 0:
                continue

            ax.plot(
                slo_mult,
                ys,
                marker=MARKERS[sys],
                color=COLORS[sys],
            )

    # --------------------------------------------------------
    #                 MAIN LOOP
    # --------------------------------------------------------
    bench_names_in_order = list(benchmarks.keys())

    for b_idx, bench_name in enumerate(bench_names_in_order):
        cfg = benchmarks[bench_name]
        print(f"[Plot] Benchmark: {bench_name}")

        curves = compute_curves_for_benchmark(
            base_dir, exp_dir_name,
            slo_analysis_script,
            cfg["rps_dirs"],
            cfg["slo_dirs"],
            cfg["slo_values"],
            cfg["base_slo"],
        )
        # print(curves)
        # for each rps, print the accuracy gap and SLO attainment gap betweeen TeDiServe and INFaaS
        print(f"INFaaS")
        for i in range(len(curves["rps_vals"]["TeDiServe"])):
            tediserve_acc = curves["overall_rps"]["TeDiServe"][i]
            INFaaS_acc = curves["overall_rps"]["INFaaS"][i]
            acc_gap = tediserve_acc - INFaaS_acc

            tediserve_slo = curves["slo_att_rps"]["TeDiServe"][i]
            INFaaS_slo = curves["slo_att_rps"]["INFaaS"][i]
            slo_gap = tediserve_slo - INFaaS_slo

            print(f"RPS {curves['rps_vals']['TeDiServe'][i]:.2f}: Acc Gap {acc_gap:.4f}, SLO Gap {slo_gap:.4f}")
        
        print(f"llumnix")
        for i in range(len(curves["rps_vals"]["TeDiServe"])):
            tediserve_acc = curves["overall_rps"]["TeDiServe"][i]
            llumnix_acc = curves["overall_rps"]["Llumnix"][i]
            acc_gap = tediserve_acc - llumnix_acc

            tediserve_slo = curves["slo_att_rps"]["TeDiServe"][i]
            llumnix_slo = curves["slo_att_rps"]["Llumnix"][i]
            slo_gap = tediserve_slo - llumnix_slo

            print(f"RPS {curves['rps_vals']['TeDiServe'][i]:.2f}: Acc Gap {acc_gap:.4f}, SLO Gap {slo_gap:.4f}")

        # for each slo, print the accuracy gap and SLO attainment gap betweeen TeDiServe and INFaaS
        print(f"INFaaS")
        for i in range(len(curves["slo_multipliers"])):
            tediserve_acc = curves["overall_slo"]["TeDiServe"][i]
            INFaaS_acc = curves["overall_slo"]["INFaaS"][i]
            acc_gap = tediserve_acc - INFaaS_acc

            tediserve_slo = curves["slo_att_slo"]["TeDiServe"][i]
            INFaaS_slo = curves["slo_att_slo"]["INFaaS"][i]
            slo_gap = tediserve_slo - INFaaS_slo

            print(f"SLO Mult {curves['slo_multipliers'][i]:.2f}: Acc Gap {acc_gap:.4f}, SLO Gap {slo_gap:.4f}")
        
        print(f"llumnix")
        for i in range(len(curves["slo_multipliers"])):
            tediserve_acc = curves["overall_slo"]["TeDiServe"][i]
            llumnix_acc = curves["overall_slo"]["Llumnix"][i]
            acc_gap = tediserve_acc - llumnix_acc

            tediserve_slo = curves["slo_att_slo"]["TeDiServe"][i]
            llumnix_slo = curves["slo_att_slo"]["Llumnix"][i]
            slo_gap = tediserve_slo - llumnix_slo

            print(f"SLO Mult {curves['slo_multipliers'][i]:.2f}: Acc Gap {acc_gap:.4f}, SLO Gap {slo_gap:.4f}")

        rps_vals = curves["rps_vals"]
        slo_multipliers = curves["slo_multipliers"]

        # Accuracy y-limits for this benchmark (if provided)
        acc_ylim = ACCURACY_YLIMS.get(bench_name, None)

        for m_i, metric in enumerate(metrics_to_plot):
            spec = METRIC_SPECS[metric]
            y_rps = curves[spec["rps_key"]]
            y_slo = curves[spec["slo_key"]]

            ax_rps = axs[m_i, 2 * b_idx]
            ax_slo = axs[m_i, 2 * b_idx + 1]

            # ----- left column (RPS / per-GPU RPS) -----
            plot_rps_curve(ax_rps, rps_vals, y_rps)
            ax_rps.grid(True, alpha=0.4)

            # Y-label only for the *left-most* RPS plots
            if b_idx == 0:
                ax_rps.set_ylabel(spec["ylabel"])
            # else: keep ticks, just no label (do nothing)

            # X-label only on bottom row
            if m_i == num_metrics - 1:
                ax_rps.set_xlabel("RPS per GPU")
            else:
                ax_rps.set_xticklabels([])

            # Save legend handles once
            if legend_handles is None:
                legend_handles, legend_labels = ax_rps.get_legend_handles_labels()

            # ----- right column (SLO multiple) -----
            plot_slo_curve(ax_slo, slo_multipliers, y_slo)
            ax_slo.grid(True, alpha=0.4)

            if m_i == num_metrics - 1:
                ax_slo.set_xlabel("SLO Multiple")
            else:
                # top row: hide x tick labels for SLO
                ax_slo.set_xticklabels([])

            # Hide y-ticks on SLO column
            ax_slo.set_yticklabels([])


            # ----- y-range logic -----
            if not spec["is_accuracy"]:
                ax_rps.set_ylim(0, 1.05)
                ax_slo.set_ylim(0, 1.05)
                # tick at 0.25
                yticks = np.arange(0, 1.01, 0.25)
                ax_rps.set_yticks(yticks)
                ax_slo.set_yticks(yticks)
            else:
                if acc_ylim is not None:
                    lo, hi = acc_ylim
                    ax_rps.set_ylim(lo, hi)
                    ax_slo.set_ylim(lo, hi)

                    # tick at 0.01
                    yticks = np.arange(
                        np.ceil(lo * 100) / 100,
                        np.floor(hi * 100) / 100 + 0.001,
                        0.01,
                    )
                    ax_rps.set_yticks(yticks)
                    ax_slo.set_yticks(yticks)
                else:
                    # fallback: auto-range from data
                    vals = []
                    for sys in SYSTEMS:
                        vals.extend([v for v in y_rps[sys] if v is not None])
                        vals.extend([v for v in y_slo[sys] if v is not None])

                    if vals:
                        mn, mx = min(vals), max(vals)
                        span = max(mx - mn, 0.02)
                        lo, hi = mn - 0.1 * span, mx + 0.1 * span
                        ax_rps.set_ylim(lo, hi)
                        ax_slo.set_ylim(lo, hi)

        # ----- SLO x-axis formatting (reverse, integer ticks only) -----
        if len(slo_multipliers) > 0:
            slo_min = min(slo_multipliers)
            slo_max = max(slo_multipliers)
            # integer ticks between min and max (inclusive), descending
            lo_int = int(np.ceil(slo_min))
            hi_int = int(np.ceil(slo_max))
            int_ticks = list(range(hi_int, lo_int - 1, -1))

            for m_i in range(num_metrics):
                ax = axs[m_i, 2 * b_idx + 1]
                ax.set_xlim(slo_max, slo_min)

                if m_i == num_metrics - 1:
                    # bottom row: show integer ticks with labels
                    ax.set_xticks(int_ticks)
                    ax.set_xticklabels([f"{t}×" for t in int_ticks])
                else:
                    # top row: no ticks, no labels
                    # ax.set_xticks([])
                    ax.set_xticklabels([])

    # --------------------------------------------------------
    #            TIGHT LAYOUT, TITLES & LEGEND
    # --------------------------------------------------------
    # Tight layout first (leaving space for legend + titles)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Need a draw for correct axis positions
    fig.canvas.draw()

    # Centered benchmark titles across each pair of columns
    for b_idx, bench_name in enumerate(bench_names_in_order):
        ax_left = axs[0, 2 * b_idx]
        ax_right = axs[0, 2 * b_idx + 1]
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()

        x_center = (pos_l.x0 + pos_r.x1) / 2.0
        y_top = max(pos_l.y1, pos_r.y1)

        fig.text(
            x_center,
            y_top + 0.02,
            bench_name,
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
        )

    # Global legend at top center
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(SYSTEMS),
            bbox_to_anchor=(0.5, 1.1),
        )

    # plt.subplots_adjust(left=0.90)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    # save as pdf
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved plot: {save_path}")


# ============================================================
#                    EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    base_dir = "/work2/10446/tchang85/stampede3/dllm"

    benchmarks = {
        "GSM8K": {
            "rps_dirs": {
                "TeDiServe": [
                    "20251127/182647", "20251127/183039", "20251127/185122",
                    "20251208/075520", "20251208/080114"
                    # "20251127/142953", "20251127/180048",
                ],
                "Llumnix": [
                    "20251127/181908", "20251127/183426", "20251127/184703",
                    "20251127/181510", "20251208/080918" #"20251127/181021",
                ],
                "INFaaS": [
                    "20251127/182256", "20251127/183818", "20251127/184316",
                    "20251127/142544", "20251208/080515" #"20251127/180516",
                ],
            },
            "slo_dirs": {
                "TeDiServe": [
                    "20251127/182647", "20251206/212506", "20251206/213519",
                    "20251206/215415", "20251206/214456",
                    # "20251127/182647", "20251127/192020", "20251127/191104",
                    # "20251127/190703", "20251127/190217",
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
                    "20251124/212006", "20251124/204610", "20251125/121423",
                    "20251125/111533", "20251125/124134",
                ],
                "Llumnix": [
                    "20251124/185130", "20251125/132058", "20251125/122827",
                    "20251125/112844", "20251125/130719",
                ],
                "INFaaS": [
                    "20251124/183458", "20251124/210259", "20251125/103451",
                    "20251125/104806", "20251125/125359",
                ],
            },
            "slo_dirs": {
                "TeDiServe": [
                    "20251124/204610", "20251125/135207", "20251125/140855",
                    "20251125/142511", "20251125/143948",
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
                    # "20251127/163006", 
                    "20251208/085456",
                    "20251127/164621", 
                    "20251127/155756",
                    # "20251127/161144", 
                    # "20251127/172514",
                    "20251208/094245",
                    "20251208/094757",
                ],
                "Llumnix": [
                    "20251127/163425", "20251127/163859", "20251127/165244",
                    "20251127/165721", "20251127/170216",
                ],
                "INFaaS": [
                    "20251127/162553", "20251127/161647", "20251127/151437",
                    "20251127/160716", "20251127/152353",
                ],
            },
            "slo_dirs": {
                "TeDiServe": [
                    # "20251127/163006", 
                    # "20251127/173528",
                    # "20251127/174048", 
                    # "20251127/174536",
                    "20251208/085456",
                    "20251208/090914",
                    "20251208/091328",
                    "20251208/093543"
                ],
                "Llumnix": ["20251127/163425"],
                "INFaaS": ["20251127/162553"],
            },
            "slo_values": [7.41, 5.928, 4.446, 2.964],
            "base_slo": 1.482,
        },
    }

    metrics = [
        "slo_attainment",
        "overall_accuracy",
    ]

    plot_multi_benchmarks(
        base_dir,
        benchmarks,
        metrics_to_plot=metrics,
        save_path="system_compare_multi.png",
    )
