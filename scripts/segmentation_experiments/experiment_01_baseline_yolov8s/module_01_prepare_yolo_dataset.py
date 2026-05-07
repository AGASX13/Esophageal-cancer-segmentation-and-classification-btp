#!/usr/bin/env python3
"""
Preprocess raw PanNuke .npy folds into YOLO segmentation labels (normalized polygons).

Filters to Esophagus patches only; maps PanNuke instance channels 0..4 to YOLO class ids.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------------------------------------------------------
# PanNuke channel semantics (instance ID per pixel; 0 = background / no instance)
# Channel index -> YOLO class id (same integer 0..4)
# Channel 5 is background aggregate in PanNuke — excluded from instance export.
# -----------------------------------------------------------------------------
PANNUKE_CELL_CLASS_NAMES: tuple[str, ...] = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)
NUM_CELL_CLASSES = len(PANNUKE_CELL_CLASS_NAMES)
BACKGROUND_CHANNEL_INDEX = 5  # 6th channel; not exported as instances

TISSUE_TARGET = "Esophagus"


@dataclass(frozen=True)
class FoldNpyBundle:
    """Resolved paths for one PanNuke fold on disk."""

    fold_dir: Path
    images_npy: Path
    masks_npy: Path
    types_npy: Path


def project_root_from_script() -> Path:
    """Assume this file lives in <root>/scripts/segmentation_experiments/experiment_01_baseline_yolov8s/."""
    return Path(__file__).resolve().parents[3]


def discover_folds(pannuke_root: Path) -> list[FoldNpyBundle]:
    """
    Find fold directories and .npy files.

    Supports:
    - fold1 / fold2 / fold3
    - fold 1 / fold 2 / fold 3  (Windows downloads often use spaces)

    Under each fold, looks for images|Images and masks|Masks folders.
    types.npy ships with PanNuke inside the images folder next to images.npy.
    """
    if not pannuke_root.is_dir():
        raise FileNotFoundError(f"PanNuke root not found: {pannuke_root}")

    fold_dirs: list[Path] = []
    for name in sorted(pannuke_root.iterdir()):
        if not name.is_dir():
            continue
        n = name.name.lower().replace(" ", "")
        if n.startswith("fold") and n[4:].isdigit():
            fold_dirs.append(name)

    if not fold_dirs:
        raise FileNotFoundError(f"No fold* directories under {pannuke_root}")

    bundles: list[FoldNpyBundle] = []
    for fd in sorted(fold_dirs, key=lambda p: p.name.lower()):
        imgs_sub = None
        for cand in ("images", "Images"):
            if (fd / cand).is_dir():
                imgs_sub = fd / cand
                break
        if imgs_sub is None:
            logging.warning("Skipping %s: no images/Images directory", fd)
            continue

        masks_sub = None
        for cand in ("masks", "Masks"):
            if (fd / cand).is_dir():
                masks_sub = fd / cand
                break
        if masks_sub is None:
            logging.warning("Skipping %s: no masks/Masks directory", fd)
            continue

        img_np = imgs_sub / "images.npy"
        typ_np = imgs_sub / "types.npy"
        msk_np = masks_sub / "masks.npy"
        if not img_np.is_file() or not typ_np.is_file() or not msk_np.is_file():
            logging.warning("Skipping %s: missing images.npy, types.npy, or masks/masks.npy", fd)
            continue

        bundles.append(
            FoldNpyBundle(
                fold_dir=fd,
                images_npy=img_np,
                masks_npy=msk_np,
                types_npy=typ_np,
            )
        )

    if not bundles:
        raise RuntimeError(f"No valid PanNuke folds with required .npy files under {pannuke_root}")
    return bundles


def fold_slug(fold_dir: Path) -> str:
    """Stable filesystem-friendly id, e.g. fold_1 from 'fold 1'."""
    return fold_dir.name.lower().replace(" ", "_")


def prepare_rgb_uint8(patch: np.ndarray) -> np.ndarray:
    """
    PanNuke images are often float64 with roughly 0–255 range (not always starting at 0).
    Convert to uint8 RGB (H, W, 3) for PNG.
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) image patch, got {patch.shape}")
    x = np.clip(patch, 0.0, 255.0)
    if x.max() <= 1.0 + 1e-6:
        x = x * 255.0
        x = np.clip(x, 0.0, 255.0)
    return np.round(x).astype(np.uint8)


def contours_to_yolo_polygon_lines(
    mask_hwc: np.ndarray,
    img_w: int,
    img_h: int,
    min_vertices: int = 3,
) -> tuple[list[str], dict[int, int]]:
    """
    For each PanNuke instance channel (0..NUM_CELL_CLASSES-1), extract external contours.

    YOLO segmentation line format (normalized):
        <class> <x1> <y1> <x2> <y2> ...

    Normalization math:
        x_norm = x_pixel / img_w
        y_norm = y_pixel / img_h
    so all coordinates lie in [0, 1] for a W x H patch (here 256 x 256).

    Returns:
        lines: list of YOLO polygon rows.
        counts_per_class: how many exported polygon objects per YOLO class id (0..4).
    """
    lines: list[str] = []
    counts_per_class: dict[int, int] = {c: 0 for c in range(NUM_CELL_CLASSES)}

    for class_id in range(NUM_CELL_CLASSES):
        # Instance label map: each unique positive integer = one nucleus instance
        ch = np.ascontiguousarray(mask_hwc[:, :, class_id])
        # PanNuke stores floats; instance ids are integers
        ch_i = np.rint(ch).astype(np.int32)
        inst_ids = np.unique(ch_i)
        inst_ids = inst_ids[inst_ids > 0]

        for inst in inst_ids:
            binary = (ch_i == int(inst)).astype(np.uint8)
            if binary.sum() == 0:
                continue

            # OpenCV expects 8-bit; foreground 255 for findContours
            binary_u8 = (binary * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cnt is None or len(cnt) < min_vertices:
                    continue
                pts = cnt.reshape(-1, 2).astype(np.float64)
                if pts.shape[0] < min_vertices:
                    continue

                flat: list[float] = []
                for px, py in pts:
                    # Normalized coordinates for YOLO (relative to patch width/height)
                    flat.append(float(px) / float(img_w))
                    flat.append(float(py) / float(img_h))

                line = str(class_id) + "".join(f" {v:.6f}" for v in flat)
                lines.append(line)
                counts_per_class[class_id] += 1

    return lines, counts_per_class


def enumerate_esophagus_patches(
    bundles: list[FoldNpyBundle],
) -> tuple[list[tuple[FoldNpyBundle, int]], int]:
    """
    Load types.npy per fold (memory-mapped where possible) and collect
    (bundle, patch_index) for TISSUE_TARGET only.

    Returns:
        eso_list: esophagus patch references.
        total_patches_all_folds: sum of all patch counts (for discarded statistic).
    """
    eso: list[tuple[FoldNpyBundle, int]] = []
    total = 0
    for b in bundles:
        types_arr = np.load(b.types_npy, allow_pickle=True)
        # Unicode strings in official folds; compare case-sensitively to 'Esophagus'
        mask_eso = types_arr == TISSUE_TARGET
        idxs = np.flatnonzero(mask_eso).tolist()
        total += int(types_arr.shape[0])
        for i in idxs:
            eso.append((b, int(i)))
    return eso, total


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(fh)
    root.addHandler(sh)


def main() -> int:
    parser = argparse.ArgumentParser(description="PanNuke -> YOLO seg (Esophagus only)")
    parser.add_argument(
        "--pannuke-root",
        type=Path,
        default=None,
        help="Default: <project>/data/raw/pannuke",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project>/data/processed/segmentation/yolo_exp1_base (creates images/{train,val} labels/{train,val})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Default: <project>/data_processing.log",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-polygon-points", type=int, default=3)
    args = parser.parse_args()

    root = project_root_from_script()
    pannuke_root = args.pannuke_root or (root / "data" / "raw" / "pannuke")
    out_root = args.output_root or (root / "data" / "processed" / "segmentation" / "yolo_exp1_base")
    log_file = args.log_file or (root / "data_processing.log")

    setup_logging(log_file)
    logging.info("Logging to %s", log_file.resolve())
    logging.info("PanNuke root: %s", pannuke_root.resolve())
    logging.info("Output root: %s", out_root.resolve())

    bundles = discover_folds(pannuke_root)
    logging.info("Discovered folds: %s", [b.fold_dir.name for b in bundles])

    eso_patches, total_all = enumerate_esophagus_patches(bundles)
    n_eso = len(eso_patches)
    n_discarded = total_all - n_eso
    logging.info("Total patches (all folds): %d", total_all)
    logging.info("Esophagus patches kept: %d", n_eso)
    logging.info("Non-Esophagus patches discarded: %d", n_discarded)

    if n_eso == 0:
        logging.error("No Esophagus patches found — check types.npy spelling (%r).", TISSUE_TARGET)
        return 1

    indices = np.arange(n_eso)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(args.val_fraction),
        random_state=int(args.seed),
        shuffle=True,
    )
    train_set = set(train_idx.tolist())
    split_name: dict[int, str] = {}
    for i in train_idx:
        split_name[int(i)] = "Train"
    for i in val_idx:
        split_name[int(i)] = "Val"

    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    # Cache open memmaps per fold to avoid reloading
    mmap_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def get_mmaps(b: FoldNpyBundle) -> tuple[np.ndarray, np.ndarray]:
        key = b.fold_dir.resolve()
        if key not in mmap_cache:
            mmap_cache[key] = (
                np.load(b.images_npy, mmap_mode="r"),
                np.load(b.masks_npy, mmap_mode="r"),
            )
        return mmap_cache[key]

    # Second tqdm pass: materialize PNG + YOLO txt
    for j in tqdm(range(n_eso), desc="Esophagus patches", unit="patch"):
        b, patch_idx = eso_patches[j]
        imgs, masks = get_mmaps(b)

        # Sanity: PanNuke stores N as first dimension — filter uses patch_idx along N
        if patch_idx >= imgs.shape[0] or patch_idx >= masks.shape[0]:
            logging.error("Index out of range fold=%s idx=%s", b.fold_dir.name, patch_idx)
            continue

        patch_img = np.array(imgs[patch_idx])  # copy from mmap for OpenCV
        patch_msk = np.array(masks[patch_idx])

        h, w = patch_img.shape[0], patch_img.shape[1]
        if patch_msk.shape[0] != h or patch_msk.shape[1] != w:
            logging.warning(
                "Mask H,W mismatch for fold=%s idx=%s: img %s mask %s",
                b.fold_dir.name,
                patch_idx,
                patch_img.shape,
                patch_msk.shape,
            )
        if patch_msk.shape[2] <= BACKGROUND_CHANNEL_INDEX:
            logging.error(
                "Expected at least %d mask channels, got shape %s",
                BACKGROUND_CHANNEL_INDEX + 1,
                patch_msk.shape,
            )
            continue

        lines, class_counts = contours_to_yolo_polygon_lines(
            patch_msk,
            img_w=w,
            img_h=h,
            min_vertices=int(args.min_polygon_points),
        )

        slug = fold_slug(b.fold_dir)
        stem = f"{slug}_idx{patch_idx:05d}"
        split = split_name[j]

        dest_img_dir = img_train if split == "Train" else img_val
        dest_lbl_dir = lbl_train if split == "Train" else lbl_val

        png_path = dest_img_dir / f"{stem}.png"
        txt_path = dest_lbl_dir / f"{stem}.txt"

        rgb = prepare_rgb_uint8(patch_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(png_path), bgr):
            logging.error("Failed to write %s", png_path)
            continue

        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        counts_str = ", ".join(
            f"{PANNUKE_CELL_CLASS_NAMES[c]}={class_counts[c]}" for c in range(NUM_CELL_CLASSES)
        )
        logging.info(
            "[%s] patch_id=%s | original_array_index=%s | split=%s | class_counts: %s",
            ts,
            stem,
            patch_idx,
            split,
            counts_str,
        )

    logging.info(
        "SUMMARY | Esophagus patches processed (written): %d | Total patches (all folds): %d | "
        "Discarded (non-Esophagus): %d | Train/Val split: %.0f%% / %.0f%% (seed=%s)",
        n_eso,
        total_all,
        n_discarded,
        100 * (1 - float(args.val_fraction)),
        100 * float(args.val_fraction),
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
