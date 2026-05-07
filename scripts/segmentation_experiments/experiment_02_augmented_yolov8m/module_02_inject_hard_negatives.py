#!/usr/bin/env python3
"""
Experiment 2 Phase 1: inject hard-negative Stroma tissue patches into the
YOLOv8 training split with empty segmentation labels.

The NCT-CRC-HE-100K Stroma .tif images are pure tissue background examples.
Adding them as empty-label training images teaches the model to suppress
false-positive cell detections on similar background morphology.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


# [EXP 2: HARD NEGATIVES] Edit this path to the Kaggle NCT-CRC-HE-100K Stroma .tif folder.
NCT_STROMA_TIF_DIR = Path("data/raw/hard_negatives/nct_crc_stroma")

# [EXP 2: HARD NEGATIVES] Output folders for the Experiment 2 YOLO training split.
DATASET_EXP2_IMAGES_TRAIN_DIR = Path("data/processed/segmentation/yolo_exp2_refined/images/train")
DATASET_EXP2_LABELS_TRAIN_DIR = Path("data/processed/segmentation/yolo_exp2_refined/labels/train")

# [EXP 2: HARD NEGATIVES] Keep these constants visible so the injection policy is easy to audit.
NUM_HARD_NEGATIVES = 35
OUTPUT_IMAGE_SIZE = (256, 256)
RANDOM_SEED = 42
OUTPUT_PREFIX = "hard_neg_STR_"


def setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)


def project_root_from_script() -> Path:
    """Assume this file lives in <root>/scripts/segmentation_experiments/experiment_02_augmented_yolov8m/."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: Path, project_root: Path) -> Path:
    """Treat relative configuration paths as relative to the project root."""
    return path if path.is_absolute() else project_root / path


def discover_tif_files(stroma_dir: Path) -> list[Path]:
    if not stroma_dir.is_dir():
        raise FileNotFoundError(
            f"NCT Stroma directory not found: {stroma_dir}. "
            "Edit NCT_STROMA_TIF_DIR at the top of this script."
        )

    tif_files = sorted(
        p for p in stroma_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if len(tif_files) < NUM_HARD_NEGATIVES:
        raise RuntimeError(
            f"Need exactly {NUM_HARD_NEGATIVES} Stroma .tif files, but found only {len(tif_files)} in {stroma_dir}"
        )
    return tif_files


def safe_output_paths(original_tif: Path, images_dir: Path, labels_dir: Path) -> tuple[Path, Path]:
    output_stem = f"{OUTPUT_PREFIX}{original_tif.stem}"
    png_path = images_dir / f"{output_stem}.png"
    txt_path = labels_dir / f"{output_stem}.txt"

    if png_path.exists() or txt_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing hard-negative output: {png_path} or {txt_path}"
        )

    return png_path, txt_path


def inject_hard_negative(original_tif: Path, png_path: Path, txt_path: Path) -> None:
    # [EXP 2: HARD NEGATIVES] Read the raw 224x224 Stroma .tif patch as an image.
    image = cv2.imread(str(original_tif), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {original_tif}")

    # [EXP 2: HARD NEGATIVES] Resize Stroma background patches to match the 256x256 YOLO dataset.
    resized = cv2.resize(image, OUTPUT_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

    # [EXP 2: HARD NEGATIVES] Save as PNG into data/processed/segmentation/yolo_exp2_refined/images/train with a traceable prefix.
    if not cv2.imwrite(str(png_path), resized):
        raise RuntimeError(f"OpenCV failed to write image: {png_path}")

    # [EXP 2: HARD NEGATIVES] Empty YOLO label means this training image contains no target cells.
    txt_path.write_text("", encoding="utf-8")


def main() -> int:
    setup_logging()

    project_root = project_root_from_script()
    stroma_dir = resolve_project_path(NCT_STROMA_TIF_DIR, project_root)
    images_dir = resolve_project_path(DATASET_EXP2_IMAGES_TRAIN_DIR, project_root)
    labels_dir = resolve_project_path(DATASET_EXP2_LABELS_TRAIN_DIR, project_root)

    logging.info("[EXP 2: HARD NEGATIVES] NCT Stroma source: %s", stroma_dir.resolve())
    logging.info("[EXP 2: HARD NEGATIVES] Output images: %s", images_dir.resolve())
    logging.info("[EXP 2: HARD NEGATIVES] Output labels: %s", labels_dir.resolve())
    logging.info(
        "[EXP 2: HARD NEGATIVES] Sampling exactly %d .tif files with seed=%d",
        NUM_HARD_NEGATIVES,
        RANDOM_SEED,
    )

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    tif_files = discover_tif_files(stroma_dir)
    rng = random.Random(RANDOM_SEED)
    selected_tifs = rng.sample(tif_files, NUM_HARD_NEGATIVES)

    injected = 0
    for original_tif in tqdm(selected_tifs, desc="Injecting hard negatives", unit="image"):
        png_path, txt_path = safe_output_paths(original_tif, images_dir, labels_dir)
        inject_hard_negative(original_tif, png_path, txt_path)
        injected += 1

    logging.info(
        "[EXP 2: HARD NEGATIVES] Injected %d hard negatives into %s and empty labels into %s",
        injected,
        images_dir.resolve(),
        labels_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
