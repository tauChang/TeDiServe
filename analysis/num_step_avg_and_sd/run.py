import json
import numpy as np
from pathlib import Path

def load_steps_from_file(path):
    """Load a JSON stats file and return a list of num_steps values."""
    with open(path, "r") as f:
        data = json.load(f)
    return [job["num_steps"] for job in data["per_job"].values()]

def compute_stats(steps):
    """Return (mean, std) for a list of step counts."""
    steps = np.array(steps)
    mean = steps.mean()
    std = steps.std(ddof=1)   # sample standard deviation
    return mean, std

def process_files(file_paths):
    """Process multiple files and print results for each."""
    for path in file_paths:
        steps = load_steps_from_file(path)
        mean, std = compute_stats(steps)
        print(f"File: {path}")
        print(f"  Jobs: {len(steps)}")
        print(f"  Mean steps: {mean:.2f}")
        print(f"  Std steps:  {std:.2f}")
        print()

# ------------------------------
# Example usage
# ------------------------------
files = [
    "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251126/155830/request_plots/avg_confidence_stats.json",
    "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251126/160633/request_plots/avg_confidence_stats.json",
    "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251126/161309/request_plots/avg_confidence_stats.json",
    "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251126/161905/request_plots/avg_confidence_stats.json",
    "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251126/171045/request_plots/avg_confidence_stats.json",
]

process_files(files)
