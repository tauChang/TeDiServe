import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt


DEFAULT_SOURCE_DIR = Path("/u/tchang85/dllm/plot/input_output_ablation/")
DEFAULT_TARGET_DIR = Path("/u/tchang85/tediserve_overleaf/figures/eval")
FIGURE_BASENAMES = ["io_ablation_compact"]


def convert_png_to_pdf(source_png: Path, destination_pdf: Path):
    image = plt.imread(source_png)
    fig, ax = plt.subplots(figsize=(image.shape[1] / 300.0, image.shape[0] / 300.0), dpi=300)
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(destination_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def export_figure(source_dir: Path, target_dir: Path, basename: str):
    source_pdf = source_dir / f"{basename}.pdf"
    source_png = source_dir / f"{basename}.png"
    target_pdf = target_dir / f"{basename}.pdf"

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
        help="Directory containing io_ablation_input/output PNG or PDF files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory where the Overleaf-ready PDFs should be written.",
    )
    args = parser.parse_args()

    for basename in FIGURE_BASENAMES:
        export_figure(args.source_dir, args.target_dir, basename)


if __name__ == "__main__":
    main()