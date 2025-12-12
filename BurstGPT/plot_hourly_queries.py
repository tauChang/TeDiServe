import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hourly_boxplot(csv_path, model="ChatGPT", log_type="Conversation log"):
    df = pd.read_csv(csv_path)
    
    # Filter the dataset
    df = df[(df["Model"] == model) & (df["Log Type"] == log_type)].copy()
    df = df.sort_values("Timestamp")

    # Add day + hour indices
    df["day"] = (df["Timestamp"] // 86400).astype(int)
    df["hour"] = ((df["Timestamp"] % 86400) // 3600).astype(int)

    # Count requests per (day, hour)
    hourly = df.groupby(["day", "hour"]).size().reset_index(name="count")

    # Pivot into: 24 lists → one per hour
    # Each list contains the counts from multiple days
    data = []
    for h in range(24):
        values = hourly[hourly["hour"] == h]["count"].tolist()
        data.append(values)

    # ------------------------------
    # Plot
    # ------------------------------
    plt.figure(figsize=(12, 5))

    b = plt.boxplot(
        data,
        showfliers=False,
        patch_artist=True,
        whis=1.5,
    )

    # Styling
    for patch in b['boxes']:
        patch.set(facecolor="#74a9cf", alpha=0.8)
    for cap in b['caps']:
        cap.set(color="gray")
    for whisker in b['whiskers']:
        whisker.set(color="gray")
    for median in b['medians']:
        median.set(color="black")

    plt.xticks(
        ticks=np.arange(1, 25),
        labels=[f"{h}h" for h in range(24)],
        rotation=45
    )

    plt.xlabel(f"{model} (Conv.)")
    plt.ylabel("Request Count")
    plt.title("Hourly Request Distribution Across Days")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("hourly_boxplot.png")


# Example usage:
plot_hourly_boxplot("data/BurstGPT_1.csv")