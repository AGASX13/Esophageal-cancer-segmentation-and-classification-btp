from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS_CSV = Path("runs/segment/esophagus_train2/results.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/evaluations/training_curves")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_columns = {
        "epoch",
        "train/box_loss",
        "val/box_loss",
        "train/seg_loss",
        "val/seg_loss",
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(M)",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Missing required columns in results CSV: "
            + ", ".join(missing_columns)
        )

    return df


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_loss_curves(df: pd.DataFrame, output_dir: Path) -> Path:
    palette = sns.color_palette("colorblind", 4)
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = df["epoch"]
    ax.plot(epochs, df["train/box_loss"], label="Train Box Loss", color=palette[0], linewidth=2.5)
    ax.plot(epochs, df["val/box_loss"], label="Validation Box Loss", color=palette[1], linewidth=2.5, linestyle="--")
    ax.plot(epochs, df["train/seg_loss"], label="Train Mask Loss", color=palette[2], linewidth=2.5)
    ax.plot(epochs, df["val/seg_loss"], label="Validation Mask Loss", color=palette[3], linewidth=2.5, linestyle="--")

    ax.set_title("YOLOv8 Segmentation Training and Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path = output_dir / "loss_curves.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved loss curves to %s", output_path)
    return output_path


def plot_map_curves(df: pd.DataFrame, output_dir: Path) -> Path:
    palette = sns.color_palette("colorblind", 2)
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = df["epoch"]
    ax.plot(epochs, df["metrics/mAP50-95(B)"], label="Box mAP50-95", color=palette[0], linewidth=2.5)
    ax.plot(epochs, df["metrics/mAP50-95(M)"], label="Mask mAP50-95", color=palette[1], linewidth=2.5)

    ax.set_title("YOLOv8 Detection and Segmentation mAP")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("mAP50-95")
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path = output_dir / "map_curves.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved mAP curves to %s", output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready YOLOv8 training curves from results.csv."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to Ultralytics YOLO results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated plots.",
    )
    return parser


def main() -> int:
    configure_logging()
    setup_plot_style()

    parser = build_parser()
    args = parser.parse_args()

    root = project_root()
    results_csv = args.results_csv if args.results_csv.is_absolute() else root / args.results_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    results_csv = results_csv.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_csv)
    plot_loss_curves(df, output_dir)
    plot_map_curves(df, output_dir)

    logging.info("Training curve generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
