#!/usr/bin/env python3
"""
Experiment 2 Phase 5: run YOLOv8 segmentation inference and apply a biological
constraint filter to remove impossible tiny nucleus hallucinations.

Filtered visualizations are saved to:
    runs/segment/exp2_filtered_predictions/
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


# [EXP 2: BIOLOGICAL FILTER] Easily editable inference routing.
MODEL_WEIGHTS = Path(
    "models/segmentation/experiment_2/"
    "yolov8m_pannuke_esophagus_exp2_augmented_best.pt"
)
INPUT_IMAGES_DIR = Path("data/processed/segmentation/yolo_exp2_augmented/images/val")
OUTPUT_DIR = Path("runs/segment/exp2_filtered_predictions")

# [EXP 2: BIOLOGICAL FILTER] Masks below this area are too small to be plausible nuclei.
MIN_NUCLEUS_AREA_PIXELS = 25
INFERENCE_IMAGE_SIZE = 256
CONFIDENCE_THRESHOLD = 0.25

CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (64, 64, 255),    # Neoplastic
    1: (64, 220, 64),    # Inflammatory
    2: (255, 170, 64),   # Connective
    3: (180, 64, 220),   # Dead
    4: (64, 220, 220),   # Epithelial
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def project_root_from_script() -> Path:
    """Assume this file lives in <root>/scripts/segmentation_experiments/experiment_02_augmented_yolov8m/."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def get_class_color(class_id: int) -> tuple[int, int, int]:
    if class_id in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[class_id]

    rng = np.random.default_rng(seed=class_id)
    color = rng.integers(low=40, high=235, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def mask_pixel_area(mask_tensor: torch.Tensor) -> int:
    # [EXP 2: BIOLOGICAL FILTER] The biological constraint is based on pixel area:
    # any predicted mask with strictly fewer than 25 foreground pixels is ignored
    # because it is too small to represent a realistic nucleus in 256x256 patches.
    mask_np = mask_tensor.detach().cpu().numpy()
    return int(np.count_nonzero(mask_np > 0.5))


def polygon_to_int_points(polygon_xy: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    points = np.asarray(polygon_xy, dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def draw_filtered_masks(
    image_bgr: np.ndarray,
    polygons_xy: list[np.ndarray],
    class_ids: list[int],
    keep_indices: list[int],
) -> np.ndarray:
    output = image_bgr.copy()
    overlay = image_bgr.copy()
    height, width = image_bgr.shape[:2]

    for idx in keep_indices:
        polygon = polygons_xy[idx]
        if polygon is None or len(polygon) < 3:
            continue

        class_id = class_ids[idx]
        color = get_class_color(class_id)
        points = polygon_to_int_points(polygon, width, height)

        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(output, [points], isClosed=True, color=color, thickness=2)
        label_anchor = tuple(points.reshape(-1, 2)[0])
        cv2.putText(
            output,
            str(class_id),
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.35, output, 0.65, 0, dst=output)
    return output


def process_result(result, output_dir: Path) -> tuple[int, int]:
    image_path = Path(result.path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        logging.warning("Skipping unreadable image: %s", image_path)
        return 0, 0

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        out_path = output_dir / f"{image_path.stem}_filtered.png"
        cv2.imwrite(str(out_path), image_bgr)
        return 0, 0

    mask_tensors = result.masks.data
    polygons_xy = result.masks.xy
    class_ids = [int(cls) for cls in result.boxes.cls.detach().cpu().numpy().tolist()]

    keep_indices: list[int] = []
    for idx, mask_tensor in enumerate(mask_tensors):
        area = mask_pixel_area(mask_tensor)
        if area >= MIN_NUCLEUS_AREA_PIXELS:
            keep_indices.append(idx)

    filtered_image = draw_filtered_masks(
        image_bgr=image_bgr,
        polygons_xy=polygons_xy,
        class_ids=class_ids,
        keep_indices=keep_indices,
    )

    out_path = output_dir / f"{image_path.stem}_filtered.png"
    if not cv2.imwrite(str(out_path), filtered_image):
        raise RuntimeError(f"Failed to write filtered visualization: {out_path}")

    return len(mask_tensors), len(keep_indices)


def main() -> int:
    setup_logging()

    project_root = project_root_from_script()
    model_weights = resolve_project_path(MODEL_WEIGHTS, project_root)
    input_dir = resolve_project_path(INPUT_IMAGES_DIR, project_root)
    output_dir = resolve_project_path(OUTPUT_DIR, project_root)

    if not model_weights.is_file():
        logging.error("Model weights not found: %s", model_weights)
        return 1
    if not input_dir.is_dir():
        logging.error("Input image directory not found: %s", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    logging.info("[EXP 2: BIOLOGICAL FILTER] Model weights: %s", model_weights.resolve())
    logging.info("[EXP 2: BIOLOGICAL FILTER] Input images: %s", input_dir.resolve())
    logging.info("[EXP 2: BIOLOGICAL FILTER] Output directory: %s", output_dir.resolve())
    logging.info(
        "[EXP 2: BIOLOGICAL FILTER] Removing masks with area < %d pixels | device=%s",
        MIN_NUCLEUS_AREA_PIXELS,
        device,
    )

    model = YOLO(str(model_weights))
    results = model.predict(
        source=str(input_dir),
        imgsz=INFERENCE_IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    images_processed = 0
    total_raw_masks = 0
    total_kept_masks = 0

    for result in tqdm(results, desc="Filtering predictions", unit="image"):
        raw_masks, kept_masks = process_result(result, output_dir)
        images_processed += 1
        total_raw_masks += raw_masks
        total_kept_masks += kept_masks

    logging.info(
        "[EXP 2: BIOLOGICAL FILTER] Complete | images=%d | raw_masks=%d | "
        "kept_masks=%d | removed_masks=%d | saved_to=%s",
        images_processed,
        total_raw_masks,
        total_kept_masks,
        total_raw_masks - total_kept_masks,
        output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
