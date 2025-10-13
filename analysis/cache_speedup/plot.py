import matplotlib.pyplot as plt

def read_latency_file(path):
    """Read a file of 'something, latency' lines and return list of latencies."""
    latencies = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    latencies.append(float(parts[1]))
                except ValueError:
                    continue  # skip bad lines
    return latencies

# Replace with your actual file paths
file1 = "TP1.txt"
file2 = "TP4.txt"

lat1 = read_latency_file(file1)
lat2 = read_latency_file(file2)

# X values = row index
steps1 = range(len(lat1))
steps2 = range(len(lat2))

plt.figure(figsize=(8, 5))
plt.plot(steps1, lat1, label="TP=1", linewidth=1.5)
plt.plot(steps2, lat2, label="TP=4", linewidth=1.5)

plt.title("Latency vs Step")
plt.xlabel("Step (row index)")
plt.ylabel("Latency (seconds)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("cache_speedup.png", dpi=300)
