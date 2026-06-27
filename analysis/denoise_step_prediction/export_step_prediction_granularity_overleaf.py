import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_FILE = Path(
    "/u/tchang85/dllm/analysis/denoise_step_prediction/granularity_tradeoff.pdf"
)
DEFAULT_TARGET_FILE = Path(
    "/u/tchang85/tediserve_overleaf/figures/tediserve/step_prediction_granularity.pdf"
)


def export_figure(source_file: Path, target_file: Path):
    if not source_file.exists():
        raise FileNotFoundError(f"Missing source figure: {source_file}")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_file)
    print(f"Copied {source_file} -> {target_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Path to the generated step-prediction granularity PDF.",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=DEFAULT_TARGET_FILE,
        help="Destination path for the Overleaf-ready step-prediction granularity PDF.",
    )
    args = parser.parse_args()

    export_figure(args.source_file, args.target_file)


if __name__ == "__main__":
    main()