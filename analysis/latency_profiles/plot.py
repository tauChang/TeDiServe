import json
import matplotlib.pyplot as plt

# Filenames
dir = "../../latency_profiles/GSAI-ML_LLaDA-8B-Base_block32/GH200"
files = ["TP1.json", "TP2.json", "TP4.json"]

# Load data
data = {}
for fname in files:
    try:
        with open(f"{dir}/{fname}", "r") as f:
            d = json.load(f)
            # convert keys to int (batch size) and values to ms
            data[fname] = {int(k): v * 1000 for k, v in d.items()}
    except FileNotFoundError:
        print(f"File {fname} not found in {dir}")

# Plot
for fname, d in data.items():
    if d:
        xs = sorted(d.keys())
        ys = [d[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=fname.split(".")[0])

plt.xlabel("Batch size")
plt.ylabel("Time (ms)")
plt.title("Batch size vs Time across TP settings")
plt.legend()
plt.grid(True)
plt.savefig("latency_profiles.png")
