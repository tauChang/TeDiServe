import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================
# 1. Compute average hourly QPS from dataset
# =============================================================
def compute_average_hourly_qps(csv_path, model="ChatGPT", log_type="Conversation log"):
    df = pd.read_csv(csv_path)

    # Filter dataset
    df = df[(df["Model"] == model) & (df["Log Type"] == log_type)].copy()
    df = df.sort_values("Timestamp")

    df["day"] = (df["Timestamp"] // 86400).astype(int)
    df["hour"] = ((df["Timestamp"] % 86400) // 3600).astype(int)

    hourly = df.groupby(["day", "hour"]).size().reset_index(name="count")

    avg_qps = []
    for h in range(24):
        counts = hourly[hourly["hour"] == h]["count"]
        qps = 0.0 if len(counts) == 0 else counts.mean() / 3600.0
        avg_qps.append(qps)

    return np.array(avg_qps)   # shape (24,)


# =============================================================
# 2. Build synthetic trace
# =============================================================
def build_synthetic_trace(avg_hourly_qps,
                          total_hours=2,
                          max_qps=None):
    """
    avg_hourly_qps: 24-length array of avg QPS per hour
    total_hours: total length of generated trace (in hours)
    max_qps: optional scaling cap
    """
    assert len(avg_hourly_qps) == 24, "Need 24-hour QPS vector"
    np.random.seed(42)

    total_minutes = total_hours * 60
    window_minutes = total_minutes / 24       # dynamic
    window_sec = window_minutes * 60

    print(f"[INFO] total_hours={total_hours}")
    print(f"[INFO] window_minutes={window_minutes:.3f}")

    # -- Build QPS curve (24 windows)
    qps_curve = avg_hourly_qps.copy()

    # Optional scaling
    if max_qps is not None:
        peak = qps_curve.max()
        if peak > 0:
            scale = max_qps / peak
            qps_curve = qps_curve * scale
            print(f"[INFO] Scaled QPS by factor {scale:.4f} (peak={peak:.4f})")

    # ----------------------------------------------------
    # Generate Poisson arrivals
    # ----------------------------------------------------
    arrival_times = []
    t = 0.0

    for i, qps in enumerate(qps_curve):
        expected = qps * window_sec
        n = np.random.poisson(expected)

        # uniform distribution inside each window (Poisson process property)
        arrivals = t + np.random.uniform(0, window_sec, size=n)
        arrival_times.extend(arrivals)

        t += window_sec

    arrival_times = np.array(sorted(arrival_times))

    print(f"[INFO] Generated {len(arrival_times)} arrivals")

    return arrival_times, qps_curve, window_minutes


# =============================================================
# 3. Visualization
# =============================================================
def visualize_trace(arrival_times, window_minutes, outdir="trace_plots"):
    os.makedirs(outdir, exist_ok=True)

    # =============================================================
    # 1. Reconstruct QPS curve from arrival_times
    # =============================================================
    window_sec = window_minutes * 60
    T = arrival_times.max()

    # number of windows
    num_windows = int(np.ceil(T / window_sec))

    qps_reconstructed = []
    window_edges = np.arange(0, (num_windows + 1) * window_sec, window_sec)

    # Count arrivals in each window
    counts, _ = np.histogram(arrival_times, bins=window_edges)

    # QPS = count / window length (seconds)
    qps_reconstructed = counts / window_sec

    # =============================================================
    # Plot reconstructed QPS curve
    # =============================================================
    for i in range(len(qps_reconstructed)):
        print(f"Window {i} (Time {i*window_minutes:.2f} min - {(i+1)*window_minutes:.2f} min): QPS = {qps_reconstructed[i]:.4f}")
    plt.figure(figsize=(12, 4))
    plt.plot(qps_reconstructed, marker="o", linewidth=1)
    plt.title("Reconstructed QPS Curve (from arrival_times)")
    plt.xlabel(f"Window Index (each = {window_minutes:.2f} min)")
    plt.ylabel("QPS")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/qps_curve_reconstructed.png", dpi=200)
    plt.close()

    # =============================================================
    # 2. Arrival scatter plot
    # =============================================================
    plt.figure(figsize=(12, 4))
    plt.scatter(arrival_times, np.zeros_like(arrival_times),
                s=5, alpha=0.4)
    plt.title("Arrival Events Over Time")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(f"{outdir}/arrival_times.png", dpi=200)
    plt.close()

    # =============================================================
    # 3. Inter-arrival histogram
    # =============================================================
    ia = np.diff(arrival_times)

    plt.figure(figsize=(8, 4))
    plt.hist(ia, bins=50, alpha=0.7)
    plt.title("Inter-Arrival Time Distribution")
    plt.xlabel("Inter-arrival time (s)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/interarrival_hist.png", dpi=200)
    plt.close()



# =============================================================
# 4. Main
# =============================================================
if __name__ == "__main__":
    total_hours = 2
    max_qps = 12
    # 1) Compute diurnal hourly pattern
    avg_qps = compute_average_hourly_qps("data/BurstGPT_1.csv")

    # 2) Build trace for X hours (change the number)
    arrival_times, qps_curve, window_minutes = build_synthetic_trace(
        avg_qps,
        total_hours=total_hours,
        max_qps=max_qps
    )

    # 3) Plot
    visualize_trace(arrival_times, window_minutes=1, outdir="trace_plots")

    print("[DONE] Synthetic trace and plots generated.")
    # write arrival_times to a file
    np.savetxt(f"burstgpt_{total_hours}hrs_{max_qps}qps.txt", arrival_times)
