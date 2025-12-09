import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <-- needed for 3D plot

def load_instances(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # data may be a list if you're appending results
    if isinstance(data, list):
        data = data[-1]   # take last entry

    instances = data["instances"]

    df = pd.DataFrame.from_dict(instances, orient="index")

    # Optional: compute total tokens
    df["total_tokens"] = df["prompt_len"] + df["expected_response_len"]

    # print latency stats
    print("Latency stats:")
    print(df["request_latency"].describe())
    
    # print prompt length stats
    print("Prompt length stats:")
    print(df["prompt_len"].describe())
    
    # print expected response length stats
    print("Expected response length stats:")
    print(df["expected_response_len"].describe())

    return df


def plot_len_vs_latency(df, prefix="plot"):
    plt.figure(figsize=(7,5))
    plt.scatter(df["prompt_len"], df["request_latency"], alpha=0.5)
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Request latency (s)")
    plt.title("Prompt length vs Latency")
    plt.grid(True)
    plt.savefig(f"{prefix}_prompt_vs_latency.png")
    plt.close()

    plt.figure(figsize=(7,5))
    plt.scatter(df["expected_response_len"], df["request_latency"], alpha=0.5)
    plt.xlabel("Response length (tokens)")
    plt.ylabel("Request latency (s)")
    plt.title("Response length vs Latency")
    plt.grid(True)
    plt.savefig(f"{prefix}_response_vs_latency.png")
    plt.close()

    plt.figure(figsize=(7,5))
    plt.scatter(df["total_tokens"], df["request_latency"], alpha=0.5)
    plt.xlabel("Total tokens (prompt + response)")
    plt.ylabel("Request latency (s)")
    plt.title("Total tokens vs Latency")
    plt.grid(True)
    plt.savefig(f"{prefix}_total_tokens_vs_latency.png")
    plt.close()


# ==========================================================
# ✅ OPTION A: 3D SCATTER PLOT
# ==========================================================
def plot_3d_prompt_response_latency(df, prefix="plot"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        df["prompt_len"],
        df["expected_response_len"],
        df["request_latency"],
        c=df["request_latency"],
        cmap="viridis",
        alpha=0.7
    )

    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Response length (tokens)")
    ax.set_zlabel("Latency (s)")
    ax.set_title("3D Plot: Prompt Len vs Response Len vs Latency")

    fig.colorbar(sc, label="Latency (s)")
    plt.savefig(f"{prefix}_3d_prompt_response_latency.png")
    plt.close()


if __name__ == "__main__":
    df = load_instances("/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251129/235050/logs/benchmark_latency_info.json")
    print(df.head())  # sanity check

    plot_len_vs_latency(df, prefix="latency_plots")
    plot_3d_prompt_response_latency(df, prefix="latency_plots")
