import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import numpy as np

def plot_qps(trace, window_seconds=5.0):
    """
    Plot QPS over time by sliding window.
    QPS(t) = (# requests arriving in [t - window, t]) / window
    """
    arrival = trace["arrival_time"].values
    T = arrival[-1]

    # Time axis for evaluation (1 second resolution)
    t_grid = np.arange(0, T + 1, 1.0)

    qps = []
    w = window_seconds

    start_idx = 0
    end_idx = 0
    N = len(arrival)

    for t in t_grid:
        # Move start pointer to drop old requests
        while start_idx < N and arrival[start_idx] < t - w:
            start_idx += 1

        # Move end pointer to include current window
        while end_idx < N and arrival[end_idx] <= t:
            end_idx += 1

        qps.append((end_idx - start_idx) / w)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(t_grid, qps, linewidth=1.2)
    plt.title(f"QPS Over Time (window = {window_seconds}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Requests per second")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qps_plot.png")


def load_and_build_trace(csv_path, model="ChatGPT", log_type="Conversation log",
                         target_hours=2):
    random.seed(41)
    df = pd.read_csv(csv_path)

    # Filter for specific model + log type
    df = df[(df["Model"] == model) & (df["Log Type"] == log_type)]

    # -------------------------------------------------------
    # 1. Identify day boundaries
    # -------------------------------------------------------
    # Assuming Timestamp is absolute seconds; multi-day means monotonic increase.
    df = df.sort_values("Timestamp")

    # Create a "day index" by dividing timestamp by 86400
    df["day"] = (df["Timestamp"] // 86400).astype(int)

    # -------------------------------------------------------
    # 2. Randomly pick a day with enough entries
    # -------------------------------------------------------
    days = df["day"].unique()
    day = random.choice(days)

    print(f"Selected day: {day}")

    df_day = df[df["day"] == day].copy()

    # -------------------------------------------------------
    # 3. Normalize timestamps to start at 0
    # -------------------------------------------------------
    t0 = df_day["Timestamp"].min()
    df_day["t_norm"] = df_day["Timestamp"] - t0

    orig_duration = df_day["t_norm"].max()    # in seconds
    print(f"Original duration for that day: {orig_duration/3600:.2f} hours")

    # -------------------------------------------------------
    # 4. Scale day → target_hours
    # -------------------------------------------------------
    target_seconds = target_hours * 3600
    scale_factor = target_seconds / orig_duration

    print(f"Scaling factor = {scale_factor:.4f}")

    df_day["arrival_time"] = df_day["t_norm"] * scale_factor

    # -------------------------------------------------------
    # 5. Build final trace dataframe
    # -------------------------------------------------------
    df_final = df_day[[
        "arrival_time",
        "Request tokens",
        "Response tokens",
        "Total tokens"
    ]].copy()

    df_final = df_final.sort_values("arrival_time").reset_index(drop=True)
    df_final["request_id"] = df_final.index

    # print rps
    total_requests = len(df_final)
    print(f"Total requests in scaled trace: {total_requests}")
    print(f"Average RPS in scaled trace: {total_requests / target_seconds:.4f}")

    return df_final

if __name__ == "__main__":
    trace = load_and_build_trace("data/BurstGPT_1.csv", model="ChatGPT", log_type="API log", target_hours=1)
    print(trace.head())
    plot_qps(trace, window_seconds=5.0)