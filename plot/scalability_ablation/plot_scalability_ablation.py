import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "scalability_128.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "scalability_ablation.png"

FAKE_PATTERN = re.compile(
    r"^(fake_executor_(\d+))\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|"
)
ACTUAL_PATTERN = re.compile(
    r"^(actual_one_token_fake_executor_(\d+))\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|"
)


def parse_summary(input_path: Path):
    actual = []
    fake = []

    with input_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            actual_match = ACTUAL_PATTERN.match(line)
            if actual_match:
                _, num_instances, avg_rt = actual_match.groups()
                actual.append((int(num_instances), float(avg_rt)))
                continue

            fake_match = FAKE_PATTERN.match(line)
            if fake_match:
                _, num_instances, avg_rt = fake_match.groups()
                fake.append((int(num_instances), float(avg_rt)))

    actual.sort(key=lambda item: item[0])
    fake.sort(key=lambda item: item[0])

    if not fake:
        raise ValueError(f"No simulator rows found in {input_path}")

    return actual, fake


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def print_relative_differences(actual, fake):
    actual_by_instances = {num_instances: avg_rt for num_instances, avg_rt in actual}
    fake_by_instances = {num_instances: avg_rt for num_instances, avg_rt in fake}

    common_instances = sorted(set(actual_by_instances) & set(fake_by_instances))
    if not common_instances:
        return

    print("Measured vs. simulation relative difference: (simulation - measured) / measured")
    for num_instances in common_instances:
        measured = actual_by_instances[num_instances]
        simulated = fake_by_instances[num_instances]
        relative_diff = (simulated - measured) / measured
        print(
            f"  instances={num_instances:<2d} measured={measured:.3f}s "
            f"simulation={simulated:.3f}s diff={relative_diff:+.2%}"
        )


def print_latency_increase_from_simulation_baseline(fake):
    fake_by_instances = {num_instances: avg_rt for num_instances, avg_rt in fake}
    baseline = fake_by_instances.get(1)
    if baseline is None:
        return

    print("Simulation latency increase relative to 1-instance simulation baseline")
    for num_instances, simulated in fake:
        relative_increase = (simulated - baseline) / baseline
        print(
            f"  instances={num_instances:<2d} simulation={simulated:.3f}s "
            f"increase={relative_increase:+.2%}"
        )


def plot_scalability(actual, fake, output_path: Path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.3,
        }
    )

    fig, ax = plt.subplots(figsize=(1.7, 1.23))

    fake_x = [item[0] for item in fake]
    fake_y = [item[1] for item in fake]
    actual_x = [item[0] for item in actual]
    actual_y = [item[1] for item in actual]

    ax.axvspan(16, max(fake_x)+10, color="#f3f3f3", alpha=0.9, zorder=0)
    # ax.axvline(16, color="#8c8c8c", linestyle="--", linewidth=1.0, zorder=1)

    fake_line = ax.plot(
        fake_x,
        fake_y,
        color="#B6771D",
        marker="o",
        markersize=3.5,
        label="Simulator",
        zorder=3,
    )

    actual_line = None
    if actual:
        actual_line = ax.plot(
            actual_x,
            actual_y,
            color="#134686",
            marker="s",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="Measured",
            zorder=4,
        )

    ax.set_xlabel("Num Instances", labelpad=1)
    ax.set_ylabel("Req Latency (s)", labelpad=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(0.9, max(fake_x) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="in", length=2)

    y_min = min(fake_y + actual_y) if actual_y else min(fake_y)
    y_max = max(fake_y + actual_y) if actual_y else max(fake_y)
    margin = max(0.4, (y_max - y_min) * 0.08)
    ax.set_ylim(y_min - margin, y_max + margin)

    top_y = y_max + margin * 0.52

    handles = []
    labels = []
    if actual_line is not None:
        handles.append(actual_line[0])
        labels.append("Measured")
    handles.append(fake_line[0])
    labels.append("Simulation")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        handlelength=1.6,
        borderpad=0.25,
        labelspacing=0.3,
    )

    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.24, top=0.92)
    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the scalability summary text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the scalability figure.",
    )
    args = parser.parse_args()

    actual, fake = parse_summary(args.input)
    print_relative_differences(actual, fake)
    print_latency_increase_from_simulation_baseline(fake)
    plot_scalability(actual, fake, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()