import re
import statistics

log_path = "log"   # path to your log file
pattern = re.compile(r"completed in ([0-9.]+)s")

times = []
with open(log_path) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            times.append(float(m.group(1)))

if times:
    print(f"Count: {len(times)}")
    print(f"Average processing time: {statistics.mean(times):.3f} s")
    print(f"Median: {statistics.median(times):.3f} s")
else:
    print("No matching lines found.")
