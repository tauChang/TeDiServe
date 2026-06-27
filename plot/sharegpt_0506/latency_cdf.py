import json
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130"
FILES = {
    "TeDiServe": "203923",
    "Llumnix": "231147",
    "InFaaS": "152122",
}
COLORS = {
    "InFaaS": "#134686",
    "Llumnix": "#B89000",
    "TeDiServe": "#FF4F0F",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 2.5,
})

def load_latencies(path):
    with open(path, "r") as f:
        data = json.load(f)

    latencies = []

    # data["instances"] is dict of dicts
    for inst_id, inst in data[0]["instances"].items():
        if "request_latency" in inst:
            latencies.append(inst["request_latency"])

    return np.array(latencies, dtype=float)


plt.figure(figsize=(8, 5))

for system, ts in [
                    ("Llumnix", FILES["Llumnix"]),
                    ("InFaaS", FILES["InFaaS"]),
                    ("TeDiServe", FILES["TeDiServe"]),
                    ]:
    json_path = os.path.join(BASE, ts, "logs", "benchmark_latency_info.json")

    print(f"Loading {system} from {json_path}")
    lat = load_latencies(json_path)

    lat_sorted = np.sort(lat)
    cdf = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)

    # plt.plot(lat_sorted, cdf, label=system, linewidth=2.0)
    plot_color = COLORS.get(system, None)
    plt.plot(lat_sorted, cdf, label=system, 
             color=plot_color, linewidth=4.0)

# Draw vertical SLO line at 10 sec
SLO = 10.0
plt.axvline(SLO, color="red", linestyle="--", linewidth=1.8)
# plt.text(SLO + 0.2, 0.05, "SLO = 10s", color="red")

plt.ylim(0.5, 1.03)
plt.xlim(0, 60)
plt.xlabel("Request Latency (seconds)")
plt.ylabel("CDF")
plt.grid(True, alpha=0.3)

order = ["TeDiServe", "Llumnix", "InFaaS"]
handles, labels = plt.gca().get_legend_handles_labels()
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: order.index(x[1]))
sorted_handles, sorted_labels = zip(*sorted_handles_labels)
plt.legend(sorted_handles, sorted_labels)

plt.tight_layout()
plt.savefig("latency_cdf.png", dpi=200, bbox_inches="tight")
plt.savefig("latency_cdf.pdf", bbox_inches="tight")
