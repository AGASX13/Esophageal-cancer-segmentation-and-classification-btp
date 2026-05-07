#!/usr/bin/env python3
"""
Experiment 2 Phase 3: medical augmentations for YOLOv8 segmentation labels.

Creates a separate augmented dataset:
    data/processed/segmentation/yolo_exp2_refined_augmented/

Validation data is copied unchanged. Training data is copied unchanged first,
then positive images receive two augmented variants with YOLO polygons warped
through Albumentations keypoints.
"""
from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


# [EXP 2: MEDICAL AUGMENTATION] Easily editable dataset routing for safe, isolated outputs.
SOURCE_DATASET_DIR = Path("data/processed/segmentation/yolo_exp2_refined")
TARGET_DATASET_DIR = Path("data/processed/segmentation/yolo_exp2_augmented")

# [EXP 2: MEDICAL AUGMENTATION] YOLO polygons were generated for 256x256 patches.
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

NUM_AUGMENTED_VERSIONS = 2
HARD_NEGATIVE_PREFIX = "hard_neg_STR_"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloPolygon:
    class_id: str
    points_norm: np.ndarray


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
    return path if path.is_absolute() else project_root / path


def build_augmentation_pipeline() -> A.Compose:
    elastic_kwargs = {
        "alpha": 120,
        "sigma": 120 * 0.05,
        "alpha_affine": 120 * 0.03,
        "p": 0.8,
    }
    try:
        elastic_transform = A.ElasticTransform(**elastic_kwargs)
    except TypeError:
        logging.warning(
            "[EXP 2: MEDICAL AUGMENTATION] Installed Albumentations version does not accept "
            "alpha_affine; continuing with alpha and sigma only."
        )
        elastic_kwargs.pop("alpha_affine")
        elastic_transform = A.ElasticTransform(**elastic_kwargs)

    return A.Compose(
        [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.8,
            ),
            elastic_transform,
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def recreate_target_dir(target_dataset: Path) -> None:
    if target_dataset.exists():
        raise FileExistsError(
            f"Target dataset already exists: {target_dataset}. "
            "Delete it manually if you want to rebuild Experiment 2 augmentations."
        )
    # [EXP 2: MEDICAL AUGMENTATION] Create a brand new target dataset so augmented
    # training artifacts never overwrite data/processed/segmentation/yolo_exp2_refined.
    target_dataset.mkdir(parents=True, exist_ok=True)


def copy_val_split(source_dataset: Path, target_dataset: Path) -> None:
    for subdir in ("images", "labels"):
        src = source_dataset / subdir / "val"
        dst = target_dataset / subdir / "val"
        if not src.is_dir():
            raise FileNotFoundError(f"Validation source directory not found: {src}")

        # [EXP 2: MEDICAL AUGMENTATION] Validation data is routed unchanged into the new dataset.
        shutil.copytree(src, dst)
        logging.info("[EXP 2: MEDICAL AUGMENTATION] Copied unchanged validation %s to %s", subdir, dst)


def train_paths(source_dataset: Path, target_dataset: Path) -> tuple[Path, Path, Path, Path]:
    src_images = source_dataset / "images" / "train"
    src_labels = source_dataset / "labels" / "train"
    dst_images = target_dataset / "images" / "train"
    dst_labels = target_dataset / "labels" / "train"

    if not src_images.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {src_images}")
    if not src_labels.is_dir():
        raise FileNotFoundError(f"Training label directory not found: {src_labels}")

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    return src_images, src_labels, dst_images, dst_labels


def list_train_images(src_images: Path) -> list[Path]:
    return sorted(p for p in src_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_yolo_label(label_path: Path) -> list[YoloPolygon]:
    polygons: list[YoloPolygon] = []
    if not label_path.exists():
        logging.warning("Missing label file; treating as empty: %s", label_path)
        return polygons

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue

        class_id = parts[0]
        values = parts[1:]
        if len(values) < 6 or len(values) % 2 != 0:
            logging.warning("Skipping malformed YOLO polygon in %s:%d", label_path, line_number)
            continue

        try:
            points = np.asarray([float(v) for v in values], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            logging.warning("Skipping non-numeric YOLO polygon in %s:%d", label_path, line_number)
            continue

        polygons.append(YoloPolygon(class_id=class_id, points_norm=np.clip(points, 0.0, 1.0)))

    return polygons


def write_yolo_label(label_path: Path, polygons: list[YoloPolygon]) -> None:
    lines: list[str] = []
    for polygon in polygons:
        flat = polygon.points_norm.reshape(-1)
        lines.append(polygon.class_id + "".join(f" {value:.6f}" for value in flat))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def polygons_to_keypoints(polygons: list[YoloPolygon]) -> tuple[list[tuple[float, float]], list[int]]:
    keypoints: list[tuple[float, float]] = []
    polygon_lengths: list[int] = []

    for polygon in polygons:
        # [EXP 2: MEDICAL AUGMENTATION] Convert YOLO normalized x/y values into absolute
        # 256x256 pixel keypoints because Albumentations transforms geometry in pixel space.
        points_abs = polygon.points_norm.copy()
        points_abs[:, 0] *= float(IMAGE_WIDTH)
        points_abs[:, 1] *= float(IMAGE_HEIGHT)

        polygon_lengths.append(points_abs.shape[0])
        keypoints.extend((float(x), float(y)) for x, y in points_abs)

    return keypoints, polygon_lengths


def keypoints_to_polygons(
    keypoints: list[tuple[float, float]],
    source_polygons: list[YoloPolygon],
    polygon_lengths: list[int],
) -> list[YoloPolygon]:
    augmented_polygons: list[YoloPolygon] = []
    cursor = 0

    for source_polygon, polygon_len in zip(source_polygons, polygon_lengths):
        points_abs = np.asarray(keypoints[cursor : cursor + polygon_len], dtype=np.float32).reshape(-1, 2)
        cursor += polygon_len

        # [EXP 2: MEDICAL AUGMENTATION] Re-normalize warped Albumentations keypoints
        # back into YOLO's 0..1 polygon format and clip boundary drift safely.
        points_norm = points_abs.copy()
        points_norm[:, 0] /= float(IMAGE_WIDTH)
        points_norm[:, 1] /= float(IMAGE_HEIGHT)
        points_norm = np.clip(points_norm, 0.0, 1.0)

        augmented_polygons.append(YoloPolygon(class_id=source_polygon.class_id, points_norm=points_norm))

    return augmented_polygons


def augment_image_and_polygons(
    image_rgb: np.ndarray,
    polygons: list[YoloPolygon],
    pipeline: A.Compose,
) -> tuple[np.ndarray, list[YoloPolygon]]:
    keypoints, polygon_lengths = polygons_to_keypoints(polygons)
    transformed = pipeline(image=image_rgb, keypoints=keypoints)
    augmented_image = transformed["image"]
    augmented_polygons = keypoints_to_polygons(
        keypoints=transformed["keypoints"],
        source_polygons=polygons,
        polygon_lengths=polygon_lengths,
    )
    return augmented_image, augmented_polygons


def copy_original_pair(src_image: Path, src_label: Path, dst_images: Path, dst_labels: Path) -> None:
    # [EXP 2: MEDICAL AUGMENTATION] Every training sample is copied first so the
    # augmented dataset contains the original Experiment 2 training distribution.
    shutil.copy2(src_image, dst_images / src_image.name)
    if src_label.exists():
        shutil.copy2(src_label, dst_labels / src_label.name)
    else:
        (dst_labels / f"{src_image.stem}.txt").write_text("", encoding="utf-8")


def process_train_split(source_dataset: Path, target_dataset: Path, pipeline: A.Compose) -> tuple[int, int, int]:
    src_images, src_labels, dst_images, dst_labels = train_paths(source_dataset, target_dataset)
    train_images = list_train_images(src_images)

    originals_copied = 0
    hard_negatives_copied = 0
    augmented_written = 0

    for src_image in tqdm(train_images, desc="Augmenting train split", unit="image"):
        src_label = src_labels / f"{src_image.stem}.txt"
        copy_original_pair(src_image, src_label, dst_images, dst_labels)
        originals_copied += 1

        if src_image.name.startswith(HARD_NEGATIVE_PREFIX):
            # [EXP 2: MEDICAL AUGMENTATION] Hard negatives are pure background controls;
            # copy them only and do not synthesize additional background variants.
            hard_negatives_copied += 1
            continue

        polygons = read_yolo_label(src_label)
        if not polygons:
            logging.warning("Skipping augmentation for empty positive-label candidate: %s", src_image)
            continue

        image_bgr = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            logging.warning("Skipping unreadable image: %s", src_image)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for aug_idx in range(1, NUM_AUGMENTED_VERSIONS + 1):
            augmented_image_rgb, augmented_polygons = augment_image_and_polygons(
                image_rgb=image_rgb,
                polygons=polygons,
                pipeline=pipeline,
            )

            out_stem = f"aug{aug_idx}_{src_image.stem}"
            out_image = dst_images / f"{out_stem}.png"
            out_label = dst_labels / f"{out_stem}.txt"

            augmented_image = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(out_image), augmented_image):
                raise RuntimeError(f"OpenCV failed to write augmented image: {out_image}")
            write_yolo_label(out_label, augmented_polygons)
            augmented_written += 1

    return originals_copied, hard_negatives_copied, augmented_written


def main() -> int:
    setup_logging()

    project_root = project_root_from_script()
    source_dataset = resolve_project_path(SOURCE_DATASET_DIR, project_root)
    target_dataset = resolve_project_path(TARGET_DATASET_DIR, project_root)

    logging.info("[EXP 2: MEDICAL AUGMENTATION] Source dataset: %s", source_dataset.resolve())
    logging.info("[EXP 2: MEDICAL AUGMENTATION] Target dataset: %s", target_dataset.resolve())

    recreate_target_dir(target_dataset)
    copy_val_split(source_dataset, target_dataset)

    pipeline = build_augmentation_pipeline()
    originals_copied, hard_negatives_copied, augmented_written = process_train_split(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        pipeline=pipeline,
    )

    logging.info(
        "[EXP 2: MEDICAL AUGMENTATION] Complete | originals_copied=%d | "
        "hard_negatives_copied_only=%d | augmented_positive_images=%d | saved_to=%s",
        originals_copied,
        hard_negatives_copied,
        augmented_written,
        target_dataset.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
