import os
import re
import subprocess
import hashlib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
#                     CACHE UTILITIES
# ============================================================
CACHE_DIR = Path(".plot_cache")
CACHE_DIR.mkdir(exist_ok=True)


def file_hash(path):
    """Hash file contents so cache invalidates when result JSON changes."""
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def cache_load(key):
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def cache_save(key, data):
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f)


# ============================================================
#                  RPS EXTRACTION (with cache)
# ============================================================
def extract_rps(experiment_dir):
    """Parse ARRIVAL_PATTERN from run_lmeval.sh and compute RPS = 1/interarrival."""
    sh_path = os.path.join(experiment_dir, "run_lmeval.sh")
    if not os.path.exists(sh_path):
        raise FileNotFoundError(f"run_lmeval.sh not found in {experiment_dir}")

    with open(sh_path, "r") as f:
        content = f.read()

    # Match first non-comment ARRIVAL_PATTERN
    m = re.search(
        r'^\s*ARRIVAL_PATTERN\s*=\s*"([0-9]+):([0-9.]+)',
        content,
        re.MULTILINE,
    )
    if not m:
        raise ValueError(f"Could not parse ARRIVAL_PATTERN in {experiment_dir}")

    interval = float(m.group(2))
    return 1.0 / interval


def extract_rps_cached(exp_dir):
    key = f"rps_{exp_dir.replace('/', '_')}"
    cached = cache_load(key)
    if cached:
        return cached["rps"]

    rps = extract_rps(exp_dir)
    cache_save(key, {"rps": rps})
    return rps


# ============================================================
#           RUN SLO ANALYSIS (cached by JSON hash + SLO)
# ============================================================
def run_slo_analysis(script_path, json_path, slo_value):
    cmd = [
        "python",
        script_path,
        "--path",
        json_path,
        "--slo",
        str(slo_value),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

    m1 = re.search(r"SLO attainment.*?: ([0-9.]+)", out)
    m3 = re.search(r"≤.*?:\s*Count:.*?Accuracy:\s*([0-9.]+)", out, re.S)

    slo_attain = float(m1.group(1)) if m1 else None
    good_acc = float(m3.group(1)) if m3 else None
    return slo_attain, good_acc


def run_slo_analysis_cached(script_path, json_path, slo_value):
    h = file_hash(json_path)
    key = f"slo_{h}_{slo_value}".replace(".", "_")

    cached = cache_load(key)
    if cached:
        return cached["att"], cached["good"]

    slo_att, good_acc = run_slo_analysis(script_path, json_path, slo_value)
    cache_save(key, {"att": slo_att, "good": good_acc, "hash": h})
    return slo_att, good_acc


# ============================================================
#                     FIND RESULT JSON
# ============================================================
def find_result_json(experiment_dir):
    results_dir = os.path.join(experiment_dir, "results")
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if fn.endswith(".json"):
                return os.path.join(root, fn)
    raise FileNotFoundError(f"No JSON result file found in {experiment_dir}")


# ============================================================
#                       MAIN PLOTTING
# ============================================================
def plot_all(
    base_dir,
    rps_dirs,
    slo_dirs,
    slo_values,
    base_slo,
    save_path="system_compare.png",
    exp_dir_name="experiment_dir",
    slo_analysis_script="analysis/slo_attainment_and_good_accuracy/run.py",
):

    # Bigger fonts / thicker lines
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 22,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "lines.linewidth": 6,
            "lines.markersize": 12,
        }
    )

    systems = ["TeDiServe", "Llumnix", "InFaas"]
    colors = {"TeDiServe": "tab:blue", "Llumnix": "tab:orange", "InFaas": "tab:green"}
    markers = {"TeDiServe": "o", "Llumnix": "s", "InFaas": "d"}

    # Storage
    rps_vals = {sys: [] for sys in systems}
    slo_attain_rps = {sys: [] for sys in systems}
    good_acc_rps = {sys: [] for sys in systems}
    slo_attain_slo = {sys: [] for sys in systems}
    good_acc_slo = {sys: [] for sys in systems}

    # ========================================================
    # LEFT COLUMN: varying RPS (cached)
    # ========================================================
    for sys in systems:
        if sys not in rps_dirs:
            continue

        for d in rps_dirs[sys]:
            exp_path = os.path.join(base_dir, exp_dir_name, d)
            print(f"[RPS] {sys}: {exp_path}")

            rps = extract_rps_cached(exp_path)
            rps_vals[sys].append(rps)

            json_path = find_result_json(exp_path)

            # Extract SLO
            with open(os.path.join(exp_path, "run_lmeval.sh")) as f:
                m = re.search(r"SLO\s*=\s*([0-9.]+)", f.read())
            exp_slo = float(m.group(1))

            slo_att, good_acc = run_slo_analysis_cached(
                os.path.join(base_dir, slo_analysis_script),
                json_path,
                exp_slo,
            )

            slo_attain_rps[sys].append(slo_att)
            good_acc_rps[sys].append(good_acc)

    # ========================================================
    # RIGHT COLUMN: varying SLO
    # ========================================================
    slo_values_sorted = sorted(slo_values, reverse=True)
    slo_multipliers = [s / base_slo for s in slo_values_sorted]

    for sys in systems:
        if sys not in slo_dirs:
            continue

        dirs_for_sys = slo_dirs[sys]

        # TeDiServe: 1 dir per SLO
        if len(dirs_for_sys) == len(slo_values_sorted):
            for d, slo in zip(dirs_for_sys, slo_values_sorted):
                exp_path = os.path.join(base_dir, exp_dir_name, d)
                json_path = find_result_json(exp_path)

                slo_att, good_acc = run_slo_analysis_cached(
                    os.path.join(base_dir, slo_analysis_script),
                    json_path,
                    slo,
                )

                slo_attain_slo[sys].append(slo_att)
                good_acc_slo[sys].append(good_acc)

        else:
            # Llumnix / InFaas: reuse single dir
            exp_path = os.path.join(base_dir, exp_dir_name, dirs_for_sys[0])
            json_path = find_result_json(exp_path)

            for slo in slo_values_sorted:
                slo_att, good_acc = run_slo_analysis_cached(
                    os.path.join(base_dir, slo_analysis_script),
                    json_path,
                    slo,
                )

                slo_attain_slo[sys].append(slo_att)
                good_acc_slo[sys].append(good_acc)

    # ========================================================
    # PLOTTING
    # ========================================================
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharex="col")

    # --------------------------------------------------------
    # LEFT TOP
    # --------------------------------------------------------
    ax = axs[0, 0]
    for sys in systems:
        if rps_vals[sys]:
            idx = sorted(range(len(rps_vals[sys])), key=lambda i: rps_vals[sys][i])
            ax.plot(
                [rps_vals[sys][i] for i in idx],
                [slo_attain_rps[sys][i] for i in idx],
                marker=markers[sys],
                color=colors[sys],
                label=sys,
            )
    ax.set_ylabel("SLO Attainment")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------------
    # LEFT BOTTOM
    # --------------------------------------------------------
    ax = axs[1, 0]
    for sys in systems:
        if rps_vals[sys]:
            idx = sorted(range(len(rps_vals[sys])), key=lambda i: rps_vals[sys][i])
            ax.plot(
                [rps_vals[sys][i] for i in idx],
                [good_acc_rps[sys][i] for i in idx],
                marker=markers[sys],
                color=colors[sys],
            )
    ax.set_ylabel("Good Accuracy")
    ax.set_xlabel("RPS")
    ax.set_ylim(0.3, 0.4)
    ax.grid(True)

    # --------------------------------------------------------
    # RIGHT TOP
    # --------------------------------------------------------
    ax = axs[0, 1]
    for sys in systems:
        if slo_attain_slo[sys]:
            ax.plot(
                slo_multipliers,
                slo_attain_slo[sys],
                marker=markers[sys],
                color=colors[sys],
            )
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # --------------------------------------------------------
    # RIGHT BOTTOM
    # --------------------------------------------------------
    ax = axs[1, 1]
    for sys in systems:
        if good_acc_slo[sys]:
            ax.plot(
                slo_multipliers,
                good_acc_slo[sys],
                marker=markers[sys],
                color=colors[sys],
            )
    ax.set_xlabel("SLO Multiple (× base SLO)")
    ax.set_ylim(0.3, 0.4)
    ax.grid(True)

    # --------------------------------------------------------
    # TICKS: every 0.5×
    # --------------------------------------------------------
    tick_min = min(slo_multipliers)
    tick_max = max(slo_multipliers)

    ticks = []
    v = tick_max
    while v >= tick_min - 1e-9:
        ticks.append(round(v, 2))
        v -= 0.5

    tick_labels = [f"{t:.1f}×" for t in ticks]

    for col in [1]:
        for row in [0, 1]:
            ax = axs[row, col]
            ax.set_xlim(tick_max, tick_min)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # --------------------------------------------------------
    # Remove top x-tick labels
    # --------------------------------------------------------
    plt.setp(axs[0, 0].get_xticklabels(), visible=False)
    plt.setp(axs[0, 1].get_xticklabels(), visible=False)

    # --------------------------------------------------------
    # Only left-column Y labels
    # --------------------------------------------------------
    plt.setp(axs[0, 1].get_yticklabels(), visible=False)
    plt.setp(axs[1, 1].get_yticklabels(), visible=False)
    axs[0, 1].set_ylabel("")
    axs[1, 1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {save_path}")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    base_dir = "/work2/10446/tchang85/stampede3/dllm"

    rps_dirs = {
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
        "InFaas": [
            "20251124/183458",
            "20251124/210259",
            "20251125/103451",
            "20251125/104806",
            "20251125/125359",
        ],
    }

    slo_dirs = {
        "TeDiServe": [
            "20251124/204610",
            "20251125/135207",
            "20251125/140855",
            "20251125/142511",
            "20251125/143948",
        ],
        "Llumnix": ["20251125/132058"],
        "InFaas": ["20251124/210259"],
    }

    base_slo = 2.4
    slo_values = [12, 9.6, 8.4, 7.2, 6]

    plot_all(
        base_dir,
        rps_dirs,
        slo_dirs,
        slo_values,
        base_slo,
        save_path="system_compare.png",
    )
