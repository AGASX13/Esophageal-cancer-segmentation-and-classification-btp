#!/usr/bin/env python3
"""
Experiment 2 Phase 1: apply Douglas-Peucker polygon smoothing to YOLOv8
segmentation labels that already exist in data/processed/segmentation/yolo_exp1_base/.

Reads:
    data/processed/segmentation/yolo_exp1_base/images/{train,val}
    data/processed/segmentation/yolo_exp1_base/labels/{train,val}

Writes:
    data/processed/segmentation/yolo_exp2_refined/images/{train,val}
    data/processed/segmentation/yolo_exp2_refined/labels/{train,val}
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_IMAGE_SIZE = 256
EPSILON_RATIO = 0.01
SPLITS = ("train", "val")


def project_root_from_script() -> Path:
    """Assume this file lives in <root>/scripts/segmentation_experiments/experiment_02_augmented_yolov8m/."""
    return Path(__file__).resolve().parents[3]


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


def copy_images(src_dataset: Path, dst_dataset: Path) -> None:
    src_images = src_dataset / "images"
    dst_images = dst_dataset / "images"

    if not src_images.is_dir():
        raise FileNotFoundError(f"Source images directory not found: {src_images}")

    if dst_images.exists():
        logging.info(
            "[EXP 2: POLYGON SMOOTHING] Replacing existing copied images directory: %s",
            dst_images,
        )
        shutil.rmtree(dst_images)

    logging.info(
        "[EXP 2: POLYGON SMOOTHING] Copying baseline images from %s to %s",
        src_images,
        dst_images,
    )
    shutil.copytree(src_images, dst_images)


def parse_yolo_segmentation_line(line: str, label_path: Path, line_number: int) -> tuple[str, np.ndarray] | None:
    parts = line.strip().split()
    if not parts:
        return None

    class_id = parts[0]
    coord_values = parts[1:]
    if len(coord_values) < 6 or len(coord_values) % 2 != 0:
        logging.warning(
            "Skipping malformed polygon in %s:%d; expected class plus x/y coordinate pairs",
            label_path,
            line_number,
        )
        return None

    try:
        coords = np.asarray([float(v) for v in coord_values], dtype=np.float32).reshape(-1, 2)
    except ValueError:
        logging.warning("Skipping non-numeric polygon in %s:%d", label_path, line_number)
        return None

    return class_id, coords


def smooth_normalized_polygon(
    normalized_points: np.ndarray,
    image_width: int,
    image_height: int,
    epsilon_ratio: float,
) -> np.ndarray:
    # [EXP 2: POLYGON SMOOTHING] YOLO labels are normalized, so first convert x/y
    # points back to absolute pixel coordinates before using OpenCV contour geometry.
    absolute_points = normalized_points.copy()
    absolute_points[:, 0] *= float(image_width)
    absolute_points[:, 1] *= float(image_height)

    contour = np.rint(absolute_points).astype(np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    epsilon = float(epsilon_ratio) * perimeter

    # [EXP 2: POLYGON SMOOTHING] Douglas-Peucker removes annotation jitter while
    # preserving the closed biological cell outline at the selected perimeter ratio.
    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    smoothed_absolute = smoothed_contour.reshape(-1, 2).astype(np.float32)

    # [EXP 2: POLYGON SMOOTHING] Convert the smoothed pixel coordinates back into
    # YOLO's normalized 0..1 format before writing the Experiment 2 label file.
    smoothed_normalized = smoothed_absolute.copy()
    smoothed_normalized[:, 0] /= float(image_width)
    smoothed_normalized[:, 1] /= float(image_height)
    return np.clip(smoothed_normalized, 0.0, 1.0)


def format_yolo_line(class_id: str, points: np.ndarray) -> str:
    flat = points.reshape(-1)
    return class_id + "".join(f" {value:.6f}" for value in flat)


def smooth_label_file(
    src_label_path: Path,
    dst_label_path: Path,
    image_width: int,
    image_height: int,
    epsilon_ratio: float,
    min_polygon_points: int,
) -> tuple[int, int, int]:
    output_lines: list[str] = []
    polygons_seen = 0
    polygons_written = 0
    polygons_skipped = 0

    for line_number, line in enumerate(src_label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = parse_yolo_segmentation_line(line, src_label_path, line_number)
        if parsed is None:
            if line.strip():
                polygons_skipped += 1
            continue

        polygons_seen += 1
        class_id, normalized_points = parsed
        smoothed_points = smooth_normalized_polygon(
            normalized_points=normalized_points,
            image_width=image_width,
            image_height=image_height,
            epsilon_ratio=epsilon_ratio,
        )

        if smoothed_points.shape[0] < min_polygon_points:
            polygons_skipped += 1
            logging.warning(
                "Skipping over-smoothed polygon in %s:%d; only %d points remain",
                src_label_path,
                line_number,
                smoothed_points.shape[0],
            )
            continue

        output_lines.append(format_yolo_line(class_id, smoothed_points))
        polygons_written += 1

    dst_label_path.parent.mkdir(parents=True, exist_ok=True)
    dst_label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return polygons_seen, polygons_written, polygons_skipped


def smooth_split_labels(
    split: str,
    src_dataset: Path,
    dst_dataset: Path,
    image_width: int,
    image_height: int,
    epsilon_ratio: float,
    min_polygon_points: int,
) -> tuple[int, int, int, int]:
    src_labels_dir = src_dataset / "labels" / split
    dst_labels_dir = dst_dataset / "labels" / split

    if not src_labels_dir.is_dir():
        raise FileNotFoundError(f"Source labels directory not found: {src_labels_dir}")

    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(src_labels_dir.glob("*.txt"))

    files_processed = 0
    polygons_seen = 0
    polygons_written = 0
    polygons_skipped = 0

    for src_label_path in tqdm(label_files, desc=f"Smoothing {split} labels", unit="file"):
        dst_label_path = dst_labels_dir / src_label_path.name
        seen, written, skipped = smooth_label_file(
            src_label_path=src_label_path,
            dst_label_path=dst_label_path,
            image_width=image_width,
            image_height=image_height,
            epsilon_ratio=epsilon_ratio,
            min_polygon_points=min_polygon_points,
        )
        files_processed += 1
        polygons_seen += seen
        polygons_written += written
        polygons_skipped += skipped

    return files_processed, polygons_seen, polygons_written, polygons_skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Experiment 2 Douglas-Peucker smoothing to existing YOLOv8 segmentation labels."
    )
    parser.add_argument("--source-dataset", type=Path, default=None, help="Default: <project>/data/processed/segmentation/yolo_exp1_base")
    parser.add_argument("--output-dataset", type=Path, default=None, help="Default: <project>/data/processed/segmentation/yolo_exp2_refined")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epsilon-ratio", type=float, default=EPSILON_RATIO)
    parser.add_argument("--min-polygon-points", type=int, default=3)
    args = parser.parse_args()

    setup_logging()

    root = project_root_from_script()
    src_dataset = args.source_dataset or (root / "data" / "processed" / "segmentation" / "yolo_exp1_base")
    dst_dataset = args.output_dataset or (root / "data" / "processed" / "segmentation" / "yolo_exp2_refined")

    logging.info("[EXP 2: POLYGON SMOOTHING] Source dataset: %s", src_dataset.resolve())
    logging.info("[EXP 2: POLYGON SMOOTHING] Output dataset: %s", dst_dataset.resolve())
    logging.info(
        "[EXP 2: POLYGON SMOOTHING] Image size: %dx%d | epsilon: %.4f * perimeter",
        args.image_width,
        args.image_height,
        args.epsilon_ratio,
    )

    copy_images(src_dataset, dst_dataset)

    total_files = 0
    total_seen = 0
    total_written = 0
    total_skipped = 0

    for split in SPLITS:
        files, seen, written, skipped = smooth_split_labels(
            split=split,
            src_dataset=src_dataset,
            dst_dataset=dst_dataset,
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            epsilon_ratio=float(args.epsilon_ratio),
            min_polygon_points=int(args.min_polygon_points),
        )
        total_files += files
        total_seen += seen
        total_written += written
        total_skipped += skipped
        logging.info(
            "[EXP 2: POLYGON SMOOTHING] Split=%s | files=%d | polygons_read=%d | "
            "polygons_written=%d | polygons_skipped=%d",
            split,
            files,
            seen,
            written,
            skipped,
        )

    logging.info(
        "[EXP 2: POLYGON SMOOTHING] Complete | files=%d | polygons_read=%d | "
        "polygons_written=%d | polygons_skipped=%d",
        total_files,
        total_seen,
        total_written,
        total_skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
