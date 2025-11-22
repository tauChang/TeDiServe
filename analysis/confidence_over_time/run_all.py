#!/usr/bin/env python3
import os
import subprocess

EXPERIMENT_ROOT = "/u/tchang85/dllm/experiment_dir"

def main():
    for day in sorted(os.listdir(EXPERIMENT_ROOT)):
        day_dir = os.path.join(EXPERIMENT_ROOT, day)
        if not os.path.isdir(day_dir):
            continue

        # Look inside day_dir for timestamp dirs
        for run in sorted(os.listdir(day_dir)):
            run_dir = os.path.join(day_dir, run)
            if not os.path.isdir(run_dir):
                continue

            step_data = os.path.join(run_dir, "step_data.json")
            workload_history = os.path.join(run_dir, "workload_history.json")

            if not (os.path.isfile(step_data) and os.path.isfile(workload_history)):
                print(f"[SKIP] {day}/{run}: missing json files")
                continue

            output_dir = os.path.join(run_dir, "request_plots_2")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                "python",
                "plot.py",
                "--step-data", step_data,
                "--workload-history", workload_history,
                "--output-dir", output_dir,
            ]

            print(f"[RUN] {day}/{run}: generating plots…")
            try:
                subprocess.run(cmd, check=True)
                print(f"[DONE] {day}/{run}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {day}/{run}: {e}")

if __name__ == "__main__":
    main()
