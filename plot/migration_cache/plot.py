import matplotlib.pyplot as plt
import numpy as np

instances = np.array([1, 2, 3, 4])
baseline = np.ones_like(instances)
actual = np.array([1.17, 1.28, 1.33, 1.38])

width = 0.35
x = np.arange(len(instances))

plt.figure(figsize=(6,4))
plt.bar(x - width/2, baseline, width, label="Ideal (1.0)", color="#CCCCCC")
plt.bar(x + width/2, actual, width, label="Observed", color="#6AA6D8")

plt.xticks(x, instances)
plt.xlabel("# Model Instances")
plt.ylabel("Normalized Latency")
plt.ylim(0.9, 1.5)
plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("normalized_latency.pdf")
plt.savefig("normalized_latency.png", dpi=300)
