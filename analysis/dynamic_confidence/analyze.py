import json
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(strategy: str):
    # ---- Paths ----
    step_data_file_path = f"step_data_dynamic_{strategy}/gsm8k_200/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json"
    eval_file_path = f"gsm8k_200/256/GSAI-ML_LLaDA-8B-Instruct_block32_{strategy}.json"

    # ---- Step data (line-delimited JSON) ----
    step_records = []
    with open(step_data_file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            step_records.append(d)

    step_df = pd.DataFrame(step_records)

    # Extract unique request order
    id_order = list(dict.fromkeys(step_df["id"]))  # preserves first appearance
    id_to_docid = {rid: i for i, rid in enumerate(id_order)}
    step_df["doc_id"] = step_df["id"].map(id_to_docid)
    step_df["strategy"] = strategy
    step_df["frac_unmasked"] = step_df["num_unmasked_tokens"] / step_df["output_length"]

    # ---- Eval data (JSON array) ----
    with open(eval_file_path, "r") as f:
        eval_data = json.load(f)

    # Each item corresponds to one example
    records = []
    for d in eval_data["samples"]["gsm8k"]:
        if d.get("filter") == "flexible-extract":
            records.append({
                "doc_id": d["doc_id"],
                "score": d.get("exact_match", None),
            })

    eval_df = pd.DataFrame(records)

    # ---- Merge ----
    # merged = step_df.merge(eval_df[["doc_id", "exact_match"]], on="doc_id", how="left")
    merged = step_df.merge(eval_df, on="doc_id", how="left")    

    return merged

def plot_doc_progress(all_data, doc_id, output_length=256, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for strategy, df in all_data.groupby("strategy"):
        doc_df = df[df["doc_id"] == doc_id]
        if doc_df.empty:
            continue

        doc_df = doc_df.sort_values("num_denoise_ran")
        plt.plot(
            doc_df["num_denoise_ran"],
            doc_df["num_unmasked_tokens"],
            label=strategy,
            linewidth=2
        )


    # Reference horizontal lines
    plt.axhline(y=0.33 * output_length, color="gray", linestyle="--", linewidth=1, label="33% unmasked")
    plt.axhline(y=0.66 * output_length, color="gray", linestyle="--", linewidth=1, label="66% unmasked")

    plt.xlabel("Denoise Step", fontsize=12)
    plt.ylabel("Num Unmasked Tokens", fontsize=12)
    plt.title(f"Doc {doc_id} — Unmasking Progress (33%, 66% marks)", fontsize=13)
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"doc_{doc_id}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def plot_aggregate_progress(all_df, output_length=256, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))

    # Compute mean and std of unmasking progress per strategy
    grouped = all_df.groupby(["strategy", "num_denoise_ran"])["num_unmasked_tokens"]
    mean_df = grouped.mean().reset_index(name="mean_unmasked")
    std_df = grouped.std().reset_index(name="std_unmasked")

    for strategy in all_df["strategy"].unique():
        mean_sub = mean_df[mean_df["strategy"] == strategy]
        std_sub = std_df[std_df["strategy"] == strategy]

        plt.plot(
            mean_sub["num_denoise_ran"],
            mean_sub["mean_unmasked"],
            label=strategy,
            linewidth=2
        )
        plt.fill_between(
            mean_sub["num_denoise_ran"],
            mean_sub["mean_unmasked"] - std_sub["std_unmasked"],
            mean_sub["mean_unmasked"] + std_sub["std_unmasked"],
            alpha=0.15,
        )

    # Add 33% / 66% horizontal lines
    plt.axhline(y=0.33 * output_length, color="gray", linestyle="--", linewidth=1, label="33% unmasked")
    plt.axhline(y=0.66 * output_length, color="gray", linestyle="--", linewidth=1, label="66% unmasked")

    plt.xlabel("Denoise Step", fontsize=12)
    plt.ylabel("Average Num Unmasked Tokens", fontsize=12)
    plt.title("Average Unmasking Progress Across 200 Docs", fontsize=13)
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "aggregate_progress.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_aggregate_progress_normalized(all_df, output_length=256, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Define common normalized x-axis (0 to 1)
    grid = np.linspace(0, 1, 100)
    strategies = sorted(all_df["strategy"].unique())

    plt.figure(figsize=(7, 5))
    avg_curves = {}

    for strategy in strategies:
        sub = all_df[all_df["strategy"] == strategy]

        # Group by document
        doc_curves = []
        for doc_id, doc_df in sub.groupby("doc_id"):
            doc_df = doc_df.sort_values("num_denoise_ran")
            steps = doc_df["num_denoise_ran"].to_numpy()
            unmasked = doc_df["num_unmasked_tokens"].to_numpy()

            if len(steps) < 2:
                continue

            total_steps = steps.max()
            normalized_x = steps / total_steps
            # interpolate to common grid
            interp = np.interp(grid, normalized_x, unmasked / output_length)
            doc_curves.append(interp)

        if not doc_curves:
            continue

        doc_curves = np.stack(doc_curves)
        mean_curve = doc_curves.mean(axis=0)
        std_curve = doc_curves.std(axis=0)
        avg_curves[strategy] = (mean_curve, std_curve)

        plt.plot(grid * 100, mean_curve * 100, label=strategy, linewidth=2)
        plt.fill_between(
            grid * 100,
            (mean_curve - std_curve) * 100,
            (mean_curve + std_curve) * 100,
            alpha=0.15,
        )

    # Add 33% and 66% reference lines
    plt.axhline(y=33, color="gray", linestyle="--", linewidth=1, label="33% unmasked")
    plt.axhline(y=66, color="gray", linestyle="--", linewidth=1, label="66% unmasked")

    plt.xlabel("Normalized Denoising Progress (%)", fontsize=12)
    plt.ylabel("Unmasked Tokens (%)", fontsize=12)
    plt.title("Normalized Average Unmasking Progress Across 200 Docs", fontsize=13)
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "aggregate_progress_normalized.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def summarize_final_performance(all_df):
    # print average num denoise steps and final accuracy per strategy
    summary = all_df.groupby("strategy").apply(
        lambda g: pd.Series({
            "avg_denoise_steps": g.groupby("doc_id")["num_denoise_ran"].max().mean(),
            "final_accuracy": g.groupby("doc_id")["score"].first().mean()
        })
    ).reset_index()
    # order by number of 5's in strategy
    summary["num_5s"] = summary["strategy"].apply(lambda s: s.count("5"))
    summary = summary.sort_values(by=["num_5s", "strategy"])
    summary = summary.drop(columns=["num_5s"])
    print(summary)

def plot_scatter(summary):
    """Scatter: Accuracy vs. Steps (colored by final-stage confidence)."""
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(
        summary["avg_denoise_steps"],
        summary["final_accuracy"],
        s=100,
        edgecolors="black",
    )
    plt.xlabel("Average denoise steps")
    plt.ylabel("Final accuracy")
    plt.title("Accuracy vs. # Steps")
    for _, row in summary.iterrows():
        plt.text(row["avg_denoise_steps"] + 0.8, row["final_accuracy"], str(row["strategy"]), fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_accuracy_vs_steps.png", dpi=200)

def plot_scatter_arrow(summary):
    """Scatter: Accuracy vs. Steps, with arrows for stage-wise confidence changes."""
    plt.figure(figsize=(6, 4))

    # --- Base scatter plot ---
    plt.scatter(
        summary["avg_denoise_steps"],
        summary["final_accuracy"],
        s=120,
        color="gray",
        edgecolors="black",
        zorder=3,
    )

    # Draw text labels
    for _, row in summary.iterrows():
        plt.text(
            row["avg_denoise_steps"] + 0.5,
            row["final_accuracy"],
            str(row["strategy"]),
            fontsize=14,
            weight="bold",
            zorder=4,
        )

    # --- Helper: build quick lookup table ---
    coords = {
        str(row["strategy"]): (row["avg_denoise_steps"], row["final_accuracy"])
        for _, row in summary.iterrows()
    }

    # --- Define arrow groups (strategy pairs) ---
    first_stage_pairs = [("555", "955"), ("559", "959"), ("595", "995"), ("599", "999")]
    second_stage_pairs = [("555", "595"), ("955", "995"), ("559", "599"), ("959", "999")]
    third_stage_pairs = [("555", "559"), ("955", "959"), ("595", "599"), ("995", "999")]

    # --- Arrow drawing helper ---
    def draw_arrows(pairs, color, label):
        for (src, dst) in pairs:
            if src not in coords or dst not in coords:
                continue
            x0, y0 = coords[src]
            x1, y1 = coords[dst]
            plt.arrow(
                x0,
                y0,
                x1 - x0,
                y1 - y0,
                color=color,
                length_includes_head=True,
                head_width=0.004,
                head_length=1.0,
                linewidth=2.5,
                alpha=0.8,
                zorder=2,
            )
        # Add one dummy handle for legend
        plt.plot([], [], color=color, label=label, linewidth=2.5)

    # --- Draw three sets with distinct colors ---
    draw_arrows(first_stage_pairs, "#e74c3c", "Vary 1st confidence")   # red
    draw_arrows(second_stage_pairs, "#3498db", "Vary 2nd confidence") # blue
    draw_arrows(third_stage_pairs, "#27ae60", "Vary 3rd confidence")   # green

    # --- Axis / Legend styling ---
    plt.xlabel("Average denoise steps", fontsize=12)
    plt.ylabel("Final accuracy", fontsize=12)
    # plt.title("Accuracy vs. # Steps", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig("scatter_accuracy_vs_steps.png", dpi=200)


def plot_bar(summary):
    """Bar: Accuracy grouped by final-stage confidence."""
    bar_df = summary.groupby("stage3")["final_accuracy"].mean().reset_index()
    plt.figure(figsize=(5, 3))
    plt.bar(bar_df["stage3"], bar_df["final_accuracy"], color="steelblue", width=0.05)
    plt.xlabel("Final-stage confidence")
    plt.ylabel("Average accuracy")
    plt.title("Accuracy vs. Final-Stage Confidence")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("bar_accuracy_vs_final_confidence.png", dpi=200)


def plot_heatmap(summary):
    """Heatmap: Accuracy by first vs. last stage confidence."""
    pivot = summary.pivot_table(values="final_accuracy", index="stage1", columns="stage3")
    plt.figure(figsize=(5, 4))
    plt.imshow(pivot, cmap="coolwarm", origin="lower", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Final-stage confidence")
    plt.ylabel("Initial-stage confidence")
    plt.title("Accuracy by Early vs. Late Confidence")
    plt.colorbar(label="Final accuracy")
    plt.tight_layout()  
    plt.savefig("heatmap_accuracy_by_confidence_stages.png", dpi=200)


def plot_tradeoff(summary):
    """Line: Tradeoff curve of accuracy vs denoise steps."""
    plt.figure(figsize=(6, 4))
    plt.plot(summary["avg_denoise_steps"], summary["final_accuracy"], "o-")
    for _, row in summary.iterrows():
        plt.text(row["avg_denoise_steps"], row["final_accuracy"], str(row["strategy"]), fontsize=8)
    plt.xlabel("Average denoise steps")
    plt.ylabel("Final accuracy")
    plt.title("Accuracy vs. Cost Across Confidence Schedules")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tradeoff_accuracy_vs_steps.png", dpi=200)

def plot_multiencoded_scatter(summary):
    """
    Scatter plot of accuracy vs. cost.
    - Color encodes first-stage confidence
    - Marker shape encodes final-stage confidence
    - Marker size encodes middle-stage confidence
    """
    plt.figure(figsize=(7, 5))

    # Define unique marker shapes for each possible final-stage confidence
    unique_shapes = summary["stage3"].unique()
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    shape_map = {val: markers[i % len(markers)] for i, val in enumerate(sorted(unique_shapes))}

    # Color map by stage1
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Normalize color scale between min and max of stage1
    norm = plt.Normalize(summary["stage1"].min(), summary["stage1"].max())

    for _, row in summary.iterrows():
        plt.scatter(
            row["avg_denoise_steps"],
            row["final_accuracy"],
            c=[cmap(norm(row["stage1"]))],
            s=150 + 200 * (row["stage2"] - summary["stage2"].min())
              / (summary["stage2"].max() - summary["stage2"].min() + 1e-8),  # encode stage2 by size
            marker=shape_map[row["stage3"]],
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label=f"Strategy {row['strategy']}"
        )
        plt.text(
            row["avg_denoise_steps"] + 0.5,
            row["final_accuracy"] + 0.002,
            str(row["strategy"]),
            fontsize=8,
        )

    # Axis labels and title
    plt.xlabel("Average denoise steps")
    plt.ylabel("Final accuracy")
    plt.title("Multi-encoded Accuracy vs Cost (Color=Stage1, Shape=Stage3, Size=Stage2)")

    # --- Legend for marker shapes (final-stage confidence) ---
    handles = [
        plt.Line2D(
            [0], [0],
            marker=shape_map[val],
            color="w",
            label=f"Final conf {val:.1f}",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
        )
        for val in sorted(unique_shapes)
    ]
    plt.legend(handles=handles, title="Final-stage confidence", loc="lower right", frameon=True)

    # --- Colorbar for first-stage confidence ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, label="First-stage confidence")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("multiencoded_scatter_accuracy_vs_steps.png", dpi=200)

def main():
    strategies = ["555", "559", "595", "599", "955", "959", "995", "999"]

    all_data = []
    for s in strategies:
        df = read_data(s)
        all_data.append(df)
        print(f"Loaded {s}, {len(df)} rows")

    all_df = pd.concat(all_data, ignore_index=True)

    # Optionally check one document
    summarize_final_performance(all_df)
    # plot_doc_progress(all_df, doc_id=1)
    # plot_aggregate_progress(all_df)
    # plot_aggregate_progress_normalized(all_df)
    # You can also loop over more:
    # for doc_id in range(5):
    #     plot_doc_progress(all_df, doc_id)

    summary = all_df.groupby("strategy").apply(
            lambda g: pd.Series({
                "avg_denoise_steps": g.groupby("doc_id")["num_denoise_ran"].max().mean(),
                "final_accuracy": g.groupby("doc_id")["score"].first().mean()
            })
    ).reset_index()
    summary[["stage1", "stage2", "stage3"]] = summary["strategy"].astype(str).apply(
        lambda s: pd.Series([int(x) / 10 for x in list(s)])
    )

    # --- Visualizations ---
    plot_scatter(summary)
    plot_scatter_arrow(summary)
    # plot_multiencoded_scatter(summary)
    # plot_bar(summary)
    # plot_heatmap(summary)
    # plot_tradeoff(summary)

if __name__ == "__main__":
    main()