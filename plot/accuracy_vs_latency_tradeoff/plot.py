import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/"

# ============================
# Directory mappings
# ============================

GSM8K_WITH_CACHE = {
    0.9: "20251126/153014",
    0.8: "20251126/153704",
    0.7: "20251126/154226",
    0.6: "20251126/154854",
    0.5: "20251126/155412",
}

GSM8K_WITHOUT_CACHE = {
    0.9: "20251126/155830",
    0.8: "20251126/160633",
    0.7: "20251126/161309",
    0.6: "20251126/161905",
    0.5: "20251126/171045",
}

MBPP_WITH_CACHE = {
    0.9: "20251126/173606",
    0.8: "20251126/174027",
    0.7: "20251126/174417",
    0.6: "20251126/174854",
    0.5: "20251126/175315",
}

MBPP_WITHOUT_CACHE = {
    0.9: "20251126/144251",
    0.8: "20251126/144748",
    0.7: "20251126/145142",
    0.6: "20251126/145639",
    0.5: "20251126/150047",
}

# ============================
# Plot settings
# ============================

MARKER_SIZE = 300  # global marker size for all scatter points

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 2.5,
})

colors = {
    0.9: "#303D2B",
    0.8: "#5B7553",
    0.7: "#8EB897",
    0.6: "#A9D0AA",
    0.5: "#E3F4E1",
}

# ============================
# Helpers
# ============================

def find_result_json(exp_dir: str) -> str:
    results_dir = os.path.join(exp_dir, "results")
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No JSON found in {exp_dir}/results/")

def load_metrics(dir_id, task_key):
    """Load accuracy + latency for GSM8K or MBPP."""
    path = find_result_json(os.path.join(BASE, dir_id))
    with open(path) as f:
        js = json.load(f)

    if task_key == "gsm8k":
        acc = js["results"]["gsm8k"]["exact_match,flexible-extract"]

    elif task_key == "mbpp":
        acc = js["results"]["mbpp_instruct"]["pass_at_1,extract_code"]

    else:
        raise ValueError(f"Unknown task key {task_key}")

    lat = js["average_response_time"]
    return acc, lat

def gather_series(mapping, task_key):
    confs, accs, lats = [], [], []
    for conf, dir_id in sorted(mapping.items(), reverse=True):
        acc, lat = load_metrics(dir_id, task_key)
        confs.append(conf)
        accs.append(acc)
        lats.append(lat)
    return confs, accs, lats

# ============================
# Load data
# ============================

gsm_wc_conf, gsm_wc_acc, gsm_wc_lat = gather_series(GSM8K_WITH_CACHE, "gsm8k")
gsm_nc_conf, gsm_nc_acc, gsm_nc_lat = gather_series(GSM8K_WITHOUT_CACHE, "gsm8k")

mbpp_wc_conf, mbpp_wc_acc, mbpp_wc_lat = gather_series(MBPP_WITH_CACHE, "mbpp")
mbpp_nc_conf, mbpp_nc_acc, mbpp_nc_lat = gather_series(MBPP_WITHOUT_CACHE, "mbpp")

# ============================
# Create Figure
# ============================

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ------------------------
# LEFT: GSM8K
# ------------------------
ax = axs[0]

ax.plot(gsm_wc_lat, gsm_wc_acc, linestyle=":", color="black", zorder=1)
ax.scatter(gsm_wc_lat, gsm_wc_acc, s=MARKER_SIZE, marker="o",
           c=[colors[c] for c in gsm_wc_conf], edgecolors="black", linewidths=2, zorder=3)

ax.plot(gsm_nc_lat, gsm_nc_acc, linestyle="--", color="dimgray", zorder=1)
ax.scatter(gsm_nc_lat, gsm_nc_acc, s=MARKER_SIZE, marker="s",
           c=[colors[c] for c in gsm_nc_conf], edgecolors="dimgray", linewidths=2, zorder=3)

ax.set_title("GSM8K", pad=14)
ax.set_xlabel("Latency (s)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.65, 0.8)
ax.grid(True, linestyle="--", alpha=0.4)

# ------------------------
# RIGHT: MBPP
# ------------------------
ax = axs[1]

ax.plot(mbpp_wc_lat, mbpp_wc_acc, linestyle=":", color="black", zorder=1)
ax.scatter(mbpp_wc_lat, mbpp_wc_acc, s=MARKER_SIZE, marker="o",
           c=[colors[c] for c in mbpp_wc_conf], edgecolors="black", linewidths=2, zorder=3)

ax.plot(mbpp_nc_lat, mbpp_nc_acc, linestyle="--", color="dimgray", zorder=1)
ax.scatter(mbpp_nc_lat, mbpp_nc_acc, s=MARKER_SIZE, marker="s",
           c=[colors[c] for c in mbpp_nc_conf], edgecolors="dimgray", linewidths=2, zorder=3)

ax.set_title("MBPP", pad=14)
ax.set_xlabel("Latency (s)")
ax.set_ylim(0.15, 0.45)
ax.grid(True, linestyle="--", alpha=0.4)
# ticks at 1, 1.5, 2
ax.set_xticks([1, 1.5, 2.0])

# ============================
# Unified Legend
# ============================

system_handles = [
    Line2D([0], [0], color="black", linestyle=":", marker="o",
           markersize=14, markerfacecolor="white", label="With Cache",
           markeredgewidth=2),
    Line2D([0], [0], color="dimgray", linestyle="--", marker="s",
           markersize=14, markerfacecolor="white", label="Without Cache",
           markeredgewidth=2),
]

conf_handles = [
    Line2D([0], [0], marker='s', linestyle='', markersize=14,
           color=colors[c], label=str(c), markeredgecolor='black', markeredgewidth=1.5)
    for c in sorted(colors.keys(), reverse=False)
]

legend1 = fig.legend(
    handles=system_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=2,
    frameon=False,
    handletextpad=1
)

# Prepend a text legend entry
conf_title_handle = Line2D(
    [0], [0],
    linestyle='',
    marker='',
    label="Confidence Threshold:"
)

conf_handles_with_title = [conf_title_handle] + conf_handles

legend2 = fig.legend(
    handles=conf_handles_with_title,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=6,                 # 1 text + 5 markers
    frameon=False,
    handletextpad=-0.3,
    columnspacing=0.2
)

fig.add_artist(legend1)


# all_handles = system_handles + conf_handles

# fig.legend(
#     handles=all_handles,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.12),
#     ncol=7,
#     frameon=False,
#     columnspacing=0.7,
#     handletextpad=0.2,  
# )


fig.tight_layout()
plt.savefig("accuracy_vs_latency_dual.png", dpi=200, bbox_inches="tight")
plt.savefig("accuracy_vs_latency_dual.pdf", bbox_inches="tight")
print("Saved: accuracy_vs_latency_dual.png")

# # ============================
# # Improved Two-Part Legend
# # ============================

# # Part 1: System Legend (cache vs no-cache)
# system_handles = [
#     Line2D([0], [0], color="black", linestyle=":", marker="o",
#            markersize=14, markerfacecolor="white", label="With Cache"),
#     Line2D([0], [0], color="dimgray", linestyle="--", marker="s",
#            markersize=14, markerfacecolor="white", label="Without Cache"),
# ]

# legend1 = fig.legend(
#     handles=system_handles,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.10),
#     ncol=2,
#     frameon=False,
# )

# # Part 2: Confidence Threshold Legend (colored squares)
# conf_handles = [
#     Line2D([0], [0], marker='s', linestyle='', markersize=14,
#            color=colors[c], label=str(c))
#     for c in sorted(colors.keys(), reverse=True)
# ]

# legend2 = fig.legend(
#     handles=conf_handles,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.20),
#     ncol=5,
#     frameon=False,
#     title="Confidence Threshold",
#     title_fontsize=20,
# )

# # Ensure legend2 doesn't overwrite legend1
# fig.add_artist(legend1)

# fig.tight_layout()
# plt.savefig("accuracy_vs_latency_dual.png", dpi=200, bbox_inches="tight")
# print("Saved: accuracy_vs_latency_dual.png")