import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_MODEL = "GSAI-ML_LLaDA-8B-Instruct"
DEFAULT_DEVICE = "GH200"
DEFAULT_TPS = ("TP1", "TP2", "TP4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cudagraph and no-cudagraph latency profiles."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model directory name under latency_profiles and latency_profiles_no_cudagraph.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device directory name under the latency profile roots.",
    )
    parser.add_argument(
        "--cudagraph-root",
        default="latency_profiles",
        help="Root directory containing cudagraph-enabled latency profiles.",
    )
    parser.add_argument(
        "--no-cudagraph-root",
        default="latency_profiles_no_cudagraph",
        help="Root directory containing no-cudagraph latency profiles.",
    )
    parser.add_argument(
        "--tps",
        nargs="+",
        default=list(DEFAULT_TPS),
        help="Tensor-parallel profile names to compare, without the .json suffix.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/latency_profiles/output",
        help="Directory where plots and summary text will be written.",
    )
    return parser.parse_args()


def load_profile(path: Path) -> dict[int, float]:
    with path.open() as handle:
        raw = json.load(handle)
    return {int(key): float(value) * 1000.0 for key, value in raw.items()}


def summarize_profile(tp: str, cudagraph: dict[int, float], no_cudagraph: dict[int, float]) -> list[str]:
    shared_tokens = sorted(set(cudagraph) & set(no_cudagraph))
    if not shared_tokens:
        return [f"{tp}: no overlapping token counts"]

    improvements = [no_cudagraph[token] - cudagraph[token] for token in shared_tokens]
    ratios = [no_cudagraph[token] / cudagraph[token] for token in shared_tokens if cudagraph[token] != 0]

    best_token = max(shared_tokens, key=lambda token: no_cudagraph[token] - cudagraph[token])
    worst_token = min(shared_tokens, key=lambda token: no_cudagraph[token] - cudagraph[token])

    return [
        f"{tp}",
        f"  overlapping_points: {len(shared_tokens)}",
        f"  average_cudagraph_ms: {sum(cudagraph[token] for token in shared_tokens) / len(shared_tokens):.4f}",
        f"  average_no_cudagraph_ms: {sum(no_cudagraph[token] for token in shared_tokens) / len(shared_tokens):.4f}",
        f"  average_delta_ms: {sum(improvements) / len(improvements):.4f}",
        f"  average_speedup_ratio: {sum(ratios) / len(ratios):.4f}",
        f"  best_delta_ms: token={best_token}, delta={no_cudagraph[best_token] - cudagraph[best_token]:.4f}",
        f"  worst_delta_ms: token={worst_token}, delta={no_cudagraph[worst_token] - cudagraph[worst_token]:.4f}",
    ]


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cudagraph_dir = repo_root / args.cudagraph_root / args.model / args.device
    no_cudagraph_dir = repo_root / args.no_cudagraph_root / args.model / args.device
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, tuple[dict[int, float], dict[int, float]]] = {}
    for tp in args.tps:
        cudagraph_path = cudagraph_dir / f"{tp}.json"
        no_cudagraph_path = no_cudagraph_dir / f"{tp}.json"
        if not cudagraph_path.is_file():
            raise FileNotFoundError(f"Missing cudagraph profile: {cudagraph_path}")
        if not no_cudagraph_path.is_file():
            raise FileNotFoundError(f"Missing no-cudagraph profile: {no_cudagraph_path}")
        profiles[tp] = (load_profile(cudagraph_path), load_profile(no_cudagraph_path))

    figure, axes = plt.subplots(
        2,
        len(args.tps),
        figsize=(6 * len(args.tps), 10),
        squeeze=False,
        sharey="row",
    )

    summary_lines = [
        f"model: {args.model}",
        f"device: {args.device}",
        "",
    ]

    latency_min = float("inf")
    latency_max = float("-inf")
    speedup_min = float("inf")
    speedup_max = float("-inf")

    for index, tp in enumerate(args.tps):
        cudagraph, no_cudagraph = profiles[tp]
        shared_tokens = sorted(set(cudagraph) & set(no_cudagraph))
        if not shared_tokens:
            raise ValueError(f"No overlapping token counts found for {tp}")

        cudagraph_values = [cudagraph[token] for token in shared_tokens]
        no_cudagraph_values = [no_cudagraph[token] for token in shared_tokens]
        speedup_values = [
            no_cudagraph[token] / cudagraph[token] if cudagraph[token] != 0 else float("inf")
            for token in shared_tokens
        ]

        latency_min = min(latency_min, min(cudagraph_values), min(no_cudagraph_values))
        latency_max = max(latency_max, max(cudagraph_values), max(no_cudagraph_values))
        finite_speedup_values = [value for value in speedup_values if value != float("inf")]
        if finite_speedup_values:
            speedup_min = min(speedup_min, min(finite_speedup_values))
            speedup_max = max(speedup_max, max(finite_speedup_values))

        latency_axis = axes[0][index]
        latency_axis.plot(shared_tokens, cudagraph_values, label="cudagraph", linewidth=2)
        latency_axis.plot(shared_tokens, no_cudagraph_values, label="no cudagraph", linewidth=2)
        latency_axis.set_title(f"{tp} Latency")
        latency_axis.set_xlabel("Scheduled tokens")
        latency_axis.set_ylabel("Latency (ms)")
        latency_axis.grid(True, alpha=0.3)
        latency_axis.legend()

        speedup_axis = axes[1][index]
        speedup_axis.plot(shared_tokens, speedup_values, color="tab:green", linewidth=2)
        speedup_axis.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
        speedup_axis.set_title(f"{tp} Speedup")
        speedup_axis.set_xlabel("Scheduled tokens")
        speedup_axis.set_ylabel("no_cudagraph / cudagraph")
        speedup_axis.grid(True, alpha=0.3)

        summary_lines.extend(summarize_profile(tp, cudagraph, no_cudagraph))
        summary_lines.append("")

    if latency_min < latency_max:
        latency_padding = (latency_max - latency_min) * 0.05
        for axis in axes[0]:
            axis.set_ylim(latency_min - latency_padding, latency_max + latency_padding)

    if speedup_min < speedup_max:
        speedup_padding = (speedup_max - speedup_min) * 0.05
        lower = min(speedup_min - speedup_padding, 1.0 - speedup_padding)
        upper = max(speedup_max + speedup_padding, 1.0 + speedup_padding)
        for axis in axes[1]:
            axis.set_ylim(lower, upper)

    figure.suptitle(f"CUDAGraph vs No-CUDAGraph Latency Comparison\n{args.model} on {args.device}")
    figure.tight_layout()

    figure_path = output_dir / f"compare_cudagraph_{args.model}_{args.device}.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    summary_path = output_dir / f"compare_cudagraph_{args.model}_{args.device}.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved figure to {figure_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()