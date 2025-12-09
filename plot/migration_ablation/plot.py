import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 4,
    "lines.markersize": 10,
})

# ===========================
# COLORS
# ===========================
COLOR_INF = "#1f77b4"   # blue
COLOR_LL  = "#d62728"   # red

# ===========================
# DATA
# ===========================
data = {
    256: {
        "rps":   [0.75, 1, 1.25, 1.5],
        "avg_inf":  [1.169, 1.2631, 1.3497, 1.4597],
        "avg_ll":   [1.1229, 1.2187, 1.3217, 1.4282],
        "p99_inf":  [1.9243, 1.9986, 2.3651, 2.7331],
        "p99_ll":   [1.6774, 1.8295, 2.0428, 2.2093],
    },

    512: {
        "rps":   [0.5, 0.625, 0.75, 0.875],
        "avg_inf":  [3.0914, 3.2677, 3.6412, 4.0143],
        "avg_ll":   [2.9874, 3.1401, 3.4464, 3.81],
        "p99_inf":  [5.2969, 5.4916, 6.5551, 6.2559],
        "p99_ll":   [4.7393, 4.9035, 5.3056, 5.8433],
    },

    1024: {
        "rps":   [0.1, 0.15, 0.2, 0.25],
        "avg_inf":  [11.17, 14.0016, 16.9, 20.8],
        "avg_ll":   [10.5414, 12.8665, 15.4173, 20.4737],
        "p99_inf":  [19.72, 25.7668, 28.54, 36.24],
        "p99_ll":   [16.8736, 21.55, 24.8792, 32.77],
    },
}

# ===========================
# PLOT
# ===========================

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col')
output_lengths = [256, 512, 1024]

for col, out_len in enumerate(output_lengths):
    d = data[out_len]

    # Compute shared y-range
    col_y_values = (
        d["avg_inf"] + d["avg_ll"] +
        d["p99_inf"] + d["p99_ll"]
    )
    ymin = min(col_y_values)
    ymax = max(col_y_values)
    pad = 0.05 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    # --------------------
    # Top Row: Mean latency
    # --------------------
    ax = axes[0, col]
    ax.plot(d["rps"], d["avg_inf"], marker='o', color=COLOR_INF)
    ax.plot(d["rps"], d["avg_ll"], marker='o', color=COLOR_LL)
    ax.set_title(f"Output Len = {out_len}", pad=15)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(ymin, ymax)

    if col == 0:
        ax.set_ylabel("Mean latency (s)")

    # --------------------
    # Bottom Row: P99 latency
    # --------------------
    ax2 = axes[1, col]
    ax2.plot(d["rps"], d["p99_inf"], marker='o', color=COLOR_INF)
    ax2.plot(d["rps"], d["p99_ll"], marker='o', color=COLOR_LL)
    ax2.set_xlabel("RPS")
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(ymin, ymax)

    if col == 0:
        ax2.set_ylabel("P99 latency (s)")


# ===========================
# GLOBAL LEGEND (bottom center)
# ===========================
handles = [
    plt.Line2D([], [], color=COLOR_INF, marker='o', label="Without Migration"),
    plt.Line2D([], [], color=COLOR_LL,  marker='o', label="With Migration")
]

fig.legend(
    handles,
    ["Without Migration", "With Migration"],
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("migration_ablation.png", bbox_inches='tight')
plt.savefig("migration_ablation.pdf", bbox_inches='tight')
