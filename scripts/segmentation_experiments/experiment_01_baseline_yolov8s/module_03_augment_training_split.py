#!/usr/bin/env python3
"""
Augment PanNuke-derived YOLO instance segmentation training data with albumentations.

Copies each original train image/label into data/processed/segmentation/yolo_exp1_augmented/, then writes 3 stochastic
augmentations per image (geometry + H&E-style color jitter). Validation set is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations import KeypointParams
from tqdm import tqdm

NUM_AUG_PER_IMAGE = 3


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def stable_int_from_stem(stem: str) -> int:
    """Deterministic per-filename integer (hash() is not stable across Python processes)."""
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 1_000_003


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(fh)
    root.addHandler(sh)


def parse_yolo_seg_lines(text: str) -> list[tuple[int, np.ndarray]]:
    """
    Parse label text into list of (class_id, points) where points is (N,2) normalized [0,1].
    """
    out: list[tuple[int, np.ndarray]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        nums = [float(x) for x in parts[1:]]
        if len(nums) % 2 != 0:
            continue
        arr = np.asarray(nums, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 3:
            continue
        out.append((cid, arr))
    return out


def denormalize_to_pixels(pts_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """YOLO norm (x/W, y/H) -> pixel coordinates (float)."""
    w, h = float(width), float(height)
    out = np.empty_like(pts_norm, dtype=np.float64)
    out[:, 0] = pts_norm[:, 0] * w
    out[:, 1] = pts_norm[:, 1] * h
    return out


def flatten_polygons_for_albumentations(
    polygons: list[tuple[int, np.ndarray]],
    width: int,
    height: int,
) -> tuple[list[tuple[float, float]], list[int], list[int]]:
    """
    Denormalize polygons to pixel xy and flatten to a single keypoint list.

    Returns:
        keypoints: list of (x, y) for albumentations (xy pixel space).
        class_ids: parallel list, one per polygon.
        lengths: vertex count per polygon (to split keypoints after transform).
    """
    keypoints: list[tuple[float, float]] = []
    class_ids: list[int] = []
    lengths: list[int] = []
    for cid, pts_n in polygons:
        pts_px = denormalize_to_pixels(pts_n, width, height)
        L = int(pts_px.shape[0])
        lengths.append(L)
        class_ids.append(cid)
        for i in range(L):
            keypoints.append((float(pts_px[i, 0]), float(pts_px[i, 1])))
    return keypoints, class_ids, lengths


def split_keypoints_to_polygons(
    keypoints: list[tuple[float, float] | list[float]],
    lengths: list[int],
) -> list[np.ndarray]:
    """Map flat transformed keypoints back to (N_i, 2) arrays (float64)."""
    out: list[np.ndarray] = []
    idx = 0
    for L in lengths:
        chunk = keypoints[idx : idx + L]
        idx += L
        arr = np.array([[float(p[0]), float(p[1])] for p in chunk], dtype=np.float64)
        out.append(arr)
    return out


def polygons_to_yolo_lines(
    class_ids: list[int],
    polygons_px: list[np.ndarray],
    width: int,
    height: int,
) -> list[str]:
    """
    Re-normalize to [0,1], clip to [0,1], drop degenerate polygons (<3 vertices).
    """
    w, h = float(width), float(height)
    lines: list[str] = []
    for cid, pts in zip(class_ids, polygons_px):
        if pts.shape[0] < 3:
            continue
        xn = np.clip(pts[:, 0] / w, 0.0, 1.0)
        yn = np.clip(pts[:, 1] / h, 0.0, 1.0)
        if np.any(~np.isfinite(xn)) or np.any(~np.isfinite(yn)):
            continue
        parts = [str(cid)]
        for xi, yi in zip(xn, yn):
            parts.append(f"{float(xi):.6f}")
            parts.append(f"{float(yi):.6f}")
        lines.append(" ".join(parts))
    return lines


def build_augmentation_pipeline(seed: int | None) -> A.Compose:
    """
    H&E-oriented pipeline. Geometric transforms update keypoints; color transforms do not.

    Mimics torchvision ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1, p=0.7)
    via RandomBrightnessContrast + HueSaturationValue inside a Sequential block with p=0.7.

    OpenCV hue is in [0, 180]; hue=0.1 in torchvision corresponds to ~18° shift scale.
    """
    # Color block applied as a unit with probability 0.7 (staining / scanner variation).
    color_block = A.Sequential(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=18,
                sat_shift_limit=20,
                val_shift_limit=0,
                p=1.0,
            ),
        ],
        p=0.7,
    )

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            color_block,
        ],
        keypoint_params=KeypointParams(format="xy", remove_invisible=False),
        seed=seed,
    )


def process_one_augmentation(
    rgb: np.ndarray,
    polygons_norm: list[tuple[int, np.ndarray]],
    *,
    seed: int,
) -> tuple[np.ndarray, list[str]] | None:
    """
    Run albumentations with fixed seed. Returns BGR image and label lines, or None if no valid polygons.
    """
    h, w = rgb.shape[:2]
    kps, cids, lengths = flatten_polygons_for_albumentations(polygons_norm, w, h)
    if not kps:
        return None

    pipe = build_augmentation_pipeline(seed=seed)
    out = pipe(image=rgb, keypoints=kps)
    aug_rgb = out["image"]
    t_kps = out["keypoints"]
    polys_px = split_keypoints_to_polygons(t_kps, lengths)
    ah, aw = aug_rgb.shape[:2]
    lines = polygons_to_yolo_lines(cids, polys_px, aw, ah)

    if not lines:
        return None
    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
    return aug_bgr, lines


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO seg train augmentation (albumentations)")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--out-images-dir", type=Path, default=None)
    parser.add_argument("--out-labels-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-aug", type=int, default=NUM_AUG_PER_IMAGE)
    args = parser.parse_args()

    root = project_root_from_script()
    images_dir = args.images_dir or (root / "data" / "processed" / "segmentation" / "yolo_exp1_base" / "images" / "train")
    labels_dir = args.labels_dir or (root / "data" / "processed" / "segmentation" / "yolo_exp1_base" / "labels" / "train")
    out_img = args.out_images_dir or (root / "data" / "processed" / "segmentation" / "yolo_exp1_augmented" / "images" / "train")
    out_lbl = args.out_labels_dir or (root / "data" / "processed" / "segmentation" / "yolo_exp1_augmented" / "labels" / "train")
    log_file = args.log_file or (root / "augmentation_processing.log")

    setup_logging(log_file)
    logging.info("Output images: %s", out_img.resolve())
    logging.info("Output labels: %s", out_lbl.resolve())

    if not images_dir.is_dir() or not labels_dir.is_dir():
        logging.error("Missing input dirs: images=%s labels=%s", images_dir, labels_dir)
        return 1

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".png")
    n_orig = len(pngs)
    logging.info("Total original training images found: %d", n_orig)

    if n_orig == 0:
        logging.error("No PNG files in %s", images_dir)
        return 1

    n_aug_written = 0
    n_aug_skipped = 0
    n_orig_written = 0
    num_aug = max(0, int(args.num_aug))

    for img_path in tqdm(pngs, desc="Augmenting train", unit="img"):
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.is_file():
            logging.warning("Missing label, skipping: %s", lbl_path)
            continue

        label_text = lbl_path.read_text(encoding="utf-8")
        polygons_norm = parse_yolo_seg_lines(label_text)

        # Always preserve the original train pair in the augmented dataset root.
        shutil.copy2(img_path, out_img / f"{stem}.png")
        shutil.copy2(lbl_path, out_lbl / f"{stem}.txt")
        n_orig_written += 1

        if not polygons_norm:
            logging.warning(
                "No valid polygons in label (originals copied only, no augs): %s",
                lbl_path,
            )
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            logging.error("Failed to read image: %s", img_path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for k in range(1, num_aug + 1):
            # Distinct seeds per image and augmentation index for reproducible diversity
            seed = int(args.base_seed) + stable_int_from_stem(stem) + k * 17_389
            result = process_one_augmentation(rgb, polygons_norm, seed=seed)
            if result is None:
                n_aug_skipped += 1
                logging.warning(
                    "Skipped aug (empty polygons after transform): %s aug%d seed=%s",
                    stem,
                    k,
                    seed,
                )
                continue
            aug_bgr, lines = result
            out_png = out_img / f"{stem}_aug{k}.png"
            out_txt = out_lbl / f"{stem}_aug{k}.txt"
            if not cv2.imwrite(str(out_png), aug_bgr):
                logging.error("Failed to write %s", out_png)
                n_aug_skipped += 1
                continue
            out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_aug_written += 1

    final_size = n_orig_written + n_aug_written
    logging.info("Original Training Images: %d", n_orig)
    logging.info("Original pairs copied to yolo_exp1_augmented: %d", n_orig_written)
    logging.info("Total Augmented Images Generated: %d", n_aug_written)
    logging.info("Augmented variants skipped (empty polygons after transform): %d", n_aug_skipped)
    logging.info("Final Training Dataset Size: %d", final_size)
    print(
        f"Summary - originals found: {n_orig}, copied: {n_orig_written}, "
        f"augmented written: {n_aug_written}, aug skipped: {n_aug_skipped}, "
        f"final train size: {final_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
