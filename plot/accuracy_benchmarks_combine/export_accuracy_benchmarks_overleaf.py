import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_SOURCE_DIR = Path("/u/tchang85/dllm/plot/accuracy_benchmarks_combine")
DEFAULT_TARGET_DIR = Path("/u/tchang85/tediserve_overleaf/figures/eval")
SOURCE_BASENAME = "system_compare_merged"
TARGET_BASENAME = "accuracy_benchmarks"


def convert_png_to_pdf(source_png: Path, destination_pdf: Path):
    image = plt.imread(source_png)
    fig, ax = plt.subplots(figsize=(image.shape[1] / 300.0, image.shape[0] / 300.0), dpi=300)
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(destination_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def export_figure(source_dir: Path, target_dir: Path):
    source_pdf = source_dir / f"{SOURCE_BASENAME}.pdf"
    source_png = source_dir / f"{SOURCE_BASENAME}.png"
    target_pdf = target_dir / f"{TARGET_BASENAME}.pdf"

    target_dir.mkdir(parents=True, exist_ok=True)

    if source_pdf.exists():
        shutil.copy2(source_pdf, target_pdf)
        print(f"Copied {source_pdf} -> {target_pdf}")
        return

    if source_png.exists():
        convert_png_to_pdf(source_png, target_pdf)
        print(f"Converted {source_png} -> {target_pdf}")
        return

    raise FileNotFoundError(f"Missing source figure: {source_pdf} or {source_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the combined accuracy benchmark PNG or PDF.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory where the Overleaf-ready PDF should be written.",
    )
    args = parser.parse_args()

    export_figure(args.source_dir, args.target_dir)


if __name__ == "__main__":
    main()