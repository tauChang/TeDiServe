import os
import re
import json
import subprocess
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
#                  CACHE UTILITIES
# ============================================================
CACHE_DIR = Path(".plot_cache")
CACHE_DIR.mkdir(exist_ok=True)


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
        c, re.MULTILINE
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
        "--slo", str(slo)
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    m_slo = re.search(r"SLO attainment.*?: ([0-9.]+)", out)
    m_good = re.search(r"Accuracy.*?: ([0-9.]+)", out)
    m_over = re.search(r"Overall accuracy.*?: ([0-9.]+)", out)

    slo_att = float(m_slo.group(1)) if m_slo else None
    good = float(m_good.group(1)) if m_good else None
    overall = float(m_over.group(1)) if m_over else None
    print(good)

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
#                     SYSTEM VISUAL CONFIG
# ============================================================
SYSTEMS = ["TeDiServe", "Llumnix", "InFaas"]
COLORS = {
    "TeDiServe": "tab:blue",
    "Llumnix": "tab:orange",
    "InFaas": "tab:green",
}
MARKERS = {
    "TeDiServe": "o",
    "Llumnix": "s",
    "InFaas": "d",
}


# ============================================================
#        COMPUTE CURVES FOR A SINGLE BENCHMARK
# ============================================================
def compute_curves_for_benchmark(
    base_dir, exp_dir_name,
    slo_analysis_script,
    rps_dirs, slo_dirs, slo_values, base_slo
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
            rps_vals[sys].append(rps)

            # read experiment SLO
            with open(os.path.join(exp_path, "run_lmeval.sh")) as f:
                m = re.search(r"SLO\s*=\s*([0-9.]+)", f.read())
            exp_slo = float(m.group(1))

            att, g, o = run_slo_analysis_cached(
                os.path.join(base_dir, slo_analysis_script),
                json_file, exp_slo
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
                    json_file, slo
                )
                slo_att_slo[sys].append(att)
                good_slo[sys].append(g)
                overall_slo[sys].append(o)

        # Case 2: one directory, sweep SLOs on the same JSON
        else:
            exp_path = os.path.join(base_dir, exp_dir_name, dirs[0])
            json_file = find_result_json(exp_path)

            for slo in slo_sorted:
                att, g, o = run_slo_analysis_cached(
                    os.path.join(base_dir, slo_analysis_script),
                    json_file, slo
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
#                     GLOBAL METRIC SPECS
# ============================================================
METRIC_SPECS = {
    "slo_attainment": {
        "rps_key": "slo_att_rps",
        "slo_key": "slo_att_slo",
        "ylabel": "SLO Attainment",
        "is_accuracy": False,
    },
    "good_accuracy": {
        "rps_key": "good_rps",
        "slo_key": "good_slo",
        "ylabel": "Good Accuracy",
        "is_accuracy": True,
    },
    "overall_accuracy": {
        "rps_key": "overall_rps",
        "slo_key": "overall_slo",
        "ylabel": "Overall Accuracy",
        "is_accuracy": True,
    },
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
    """Fully modular, clean plotting of all benchmarks."""
    # Validate metrics
    for m in metrics_to_plot:
        if m not in METRIC_SPECS:
            raise ValueError(f"Invalid metric '{m}'")

    # Global style
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 20,
        "lines.linewidth": 4,
        "lines.markersize": 10,
    })

    num_bench = len(benchmarks)
    num_metrics = len(metrics_to_plot)

    fig, axs = plt.subplots(
        num_metrics,
        2 * num_bench,
        figsize=(9 * num_bench, 4 * num_metrics),
    )

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

            ax.plot(xs_sorted, ys_sorted,
                    marker=MARKERS[sys], color=COLORS[sys], label=sys)

    def plot_slo_curve(ax, slo_mult, ydict):
        for sys in SYSTEMS:
            ys = ydict[sys]
            ys = [y for y in ys if y is not None]
            if len(ys) == 0:
                continue

            ax.plot(slo_mult, ys,
                    marker=MARKERS[sys], color=COLORS[sys])

    # --------------------------------------------------------
    #                 MAIN LOOP
    # --------------------------------------------------------
    for b_idx, (bench_name, cfg) in enumerate(benchmarks.items()):
        print(f"[Plot] Benchmark: {bench_name}")

        curves = compute_curves_for_benchmark(
            base_dir, exp_dir_name,
            slo_analysis_script,
            cfg["rps_dirs"],
            cfg["slo_dirs"],
            cfg["slo_values"],
            cfg["base_slo"],
        )

        rps_vals = curves["rps_vals"]
        slo_multipliers = curves["slo_multipliers"]

        for m_i, metric in enumerate(metrics_to_plot):
            spec = METRIC_SPECS[metric]
            y_rps = curves[spec["rps_key"]]
            y_slo = curves[spec["slo_key"]]

            ax_rps = axs[m_i, 2 * b_idx]
            ax_slo = axs[m_i, 2 * b_idx + 1]

            # ----- plot left (RPS) -----
            plot_rps_curve(ax_rps, rps_vals, y_rps)
            ax_rps.grid(True)

            # Title only on top metric row
            if m_i == 0:
                ax_rps.set_title(bench_name, pad=20)

            # Legend only once
            if legend_handles is None:
                legend_handles, legend_labels = ax_rps.get_legend_handles_labels()

            # y-label on left column only
            ax_rps.set_ylabel(spec["ylabel"])

            if m_i == num_metrics - 1:
                ax_rps.set_xlabel("RPS")
            else:
                ax_rps.set_xticklabels([])

            # ----- plot right (SLO mult) -----
            plot_slo_curve(ax_slo, slo_multipliers, y_slo)
            ax_slo.grid(True)

            if m_i == num_metrics - 1:
                ax_slo.set_xlabel("SLO Multiple")
            else:
                ax_slo.set_xticklabels([])

            # Hide y ticks on right
            ax_slo.set_yticklabels([])

            # ----- y-range logic -----
            if not spec["is_accuracy"]:
                ax_rps.set_ylim(0, 1.05)
                ax_slo.set_ylim(0, 1.05)
            else:
                # Accuracy-like: align within benchmark
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

        # ----- SLO x-axis formatting (inversion) -----
        if len(slo_multipliers) > 0:
            slo_sorted = sorted(slo_multipliers, reverse=True)
            for m_i in range(num_metrics):
                ax = axs[m_i, 2 * b_idx + 1]
                ax.set_xlim(max(slo_sorted), min(slo_sorted))
                ax.set_xticks(slo_sorted)
                ax.set_xticklabels([f"{s:.1f}×" for s in slo_sorted])

    # --------------------------------------------------------
    #                      GLOBAL LEGEND
    # --------------------------------------------------------
    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=len(SYSTEMS),
            bbox_to_anchor=(0.5, 1.02)
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
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
                    "20251127/142953", "20251127/180048"
                ],
                "Llumnix": [
                    "20251127/181908", "20251127/183426", "20251127/184703",
                    "20251127/181510", "20251127/181021"
                ],
                "InFaas": [
                    "20251127/182256", "20251127/183818", "20251127/184316",
                    "20251127/142544", "20251127/180516"
                ],
            },
            "slo_dirs": {
                "TeDiServe": [
                    "20251127/182647", "20251127/192020", "20251127/191104",
                    "20251127/190703", "20251127/190217"
                ],
                "Llumnix": ["20251127/181908"],
                "InFaas": ["20251127/182256"],
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
                "InFaas": [
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
                "InFaas": ["20251124/210259"],
            },
            "slo_values": [12, 9.6, 8.4, 7.2, 6],
            "base_slo": 2.4,
        },
        "MBPP": {
            "rps_dirs": {
                "TeDiServe": [
                    "20251127/163006", "20251127/164621", "20251127/155756",
                    "20251127/161144", "20251127/172514"
                ],
                "Llumnix": [
                    "20251127/163425", "20251127/163859", "20251127/165244",
                    "20251127/165721", "20251127/170216"
                ],
                "InFaas": [
                    "20251127/162553", "20251127/161647", "20251127/151437",
                    "20251127/160716", "20251127/152353"
                ],
            },
            "slo_dirs": {
                "TeDiServe": [
                    "20251127/163006", "20251127/173528", "20251127/174048",
                    "20251127/174536"
                ],
                "Llumnix": ["20251127/163425"],
                "InFaas": ["20251127/162553"],
            },
            "slo_values": [7.41, 5.928, 4.446, 2.964],
            "base_slo": 1.482,
        }
    }

    metrics = [
        "slo_attainment", 
        # "good_accuracy", 
        "overall_accuracy"]

    plot_multi_benchmarks(
        base_dir,
        benchmarks,
        metrics_to_plot=metrics,
        save_path="system_compare_multi.png",
    )

# if __name__ == "__main__":
#     base_dir = "/work2/10446/tchang85/stampede3/dllm"

#     benchmarks = {
#         "GSM8K": {
#             "rps_dirs": {
#                 "TeDiServe": [
#                     "20251124/142409", "20251124/140346", "20251124/142010",
#                     "20251124/140920", "20251124/141428",
#                 ],
#                 "Llumnix": [
#                     "20251124/100549", "20251124/123142", "20251124/101929",
#                     "20251124/115427", "20251124/110143",
#                 ],
#                 "InFaas": [
#                     "20251124/100120", "20251124/123628", "20251124/101514",
#                     "20251124/114821", "20251124/105552",
#                 ],
#             },
#             "slo_dirs": {
#                 "TeDiServe": [
#                     "20251124/143108", "20251124/143601", "20251124/144018",
#                     "20251124/144414", "20251124/144818",
#                 ],
#                 "Llumnix": ["20251124/123142"],
#                 "InFaas": ["20251124/123628"],
#             },
#             "slo_values": [4.8, 4.2, 3.6, 3.0, 2.4],
#             "base_slo": 2.4,
#         },

#         "MMLU-Pro": {
#             "rps_dirs": {
#                 "TeDiServe": [
#                     "20251124/212006", "20251124/204610", "20251125/121423",
#                     "20251125/111533", "20251125/124134",
#                 ],
#                 "Llumnix": [
#                     "20251124/185130", "20251125/132058", "20251125/122827",
#                     "20251125/112844", "20251125/130719",
#                 ],
#                 "InFaas": [
#                     "20251124/183458", "20251124/210259", "20251125/103451",
#                     "20251125/104806", "20251125/125359",
#                 ],
#             },
#             "slo_dirs": {
#                 "TeDiServe": [
#                     "20251124/204610", "20251125/135207", "20251125/140855",
#                     "20251125/142511", "20251125/143948",
#                 ],
#                 "Llumnix": ["20251125/132058"],
#                 "InFaas": ["20251124/210259"],
#             },
#             "slo_values": [12, 9.6, 8.4, 7.2, 6],
#             "base_slo": 2.4,
#         },

#         "MBPP": {
#             "rps_dirs": {
#                 "TeDiServe": [
#                     "20251126/130954", "20251126/132100", "20251126/133333",
#                     "20251126/134643", "20251126/140246",
#                 ],
#                 "Llumnix": [
#                     "20251126/131320", "20251126/132520", "20251126/133849",
#                     "20251126/135020", "20251126/135432",
#                 ],
#                 "InFaas": [
#                     "20251126/125934", "20251126/131709", "20251126/132919",
#                     "20251126/134305", "20251126/135857",
#                 ],
#             },
#             "slo_dirs": {
#                 "TeDiServe": [
#                     "20251126/132100", "20251126/151434",
#                     "20251126/151917", "20251126/152413",
#                 ],
#                 "Llumnix": ["20251126/132520"],
#                 "InFaas": ["20251126/131709"],
#             },
#             "slo_values": [9.24, 7.392, 5.544, 3.696],
#             "base_slo": 1.848,
#         },
#     }

#     metrics = ["slo_attainment", "good_accuracy", "overall_accuracy"]

#     plot_multi_benchmarks(
#         base_dir,
#         benchmarks,
#         metrics_to_plot=metrics,
#         save_path="system_compare_multi.png",
#     )
