#!/usr/bin/env python3
"""
Visual sanity check: overlay YOLO segmentation polygons on training images.

Randomly samples PNGs from data/processed/segmentation/yolo_exp1_base/images/train/, parses matching labels,
denormalizes coordinates to pixels, draws polylines per class with distinct colors.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES: tuple[str, ...] = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)

# BGR — bright, distinguishable on H&E-style patches
CLASS_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),  # 0 Neoplastic — red
    (255, 255, 0),  # 1 Inflammatory — cyan (BGR)
    (0, 255, 0),  # 2 Connective — green
    (255, 0, 255),  # 3 Dead — magenta
    (0, 165, 255),  # 4 Epithelial — orange
)


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_yolo_seg_line(line: str) -> tuple[int, np.ndarray] | None:
    """
    One YOLO segmentation line: class x1 y1 x2 y2 ...
    Coordinates normalized in [0, 1].
    Returns (class_id, points_Nx2 float64 in normalized space) or None if invalid.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 7:  # class + at least 3 (x,y) pairs
        return None
    try:
        cid = int(parts[0])
    except ValueError:
        return None
    nums = [float(x) for x in parts[1:]]
    if len(nums) % 2 != 0:
        return None
    pts = np.array(nums, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    return cid, pts


def denormalize_polygon(pts_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    YOLO uses x / W and y / H. Map back to pixel coordinates for cv2.

    Pixel x = x_norm * width, pixel y = y_norm * height.
    Clip to image bounds for safe drawing.
    """
    w, h = float(width), float(height)
    out = np.empty_like(pts_norm, dtype=np.int32)
    out[:, 0] = np.clip(np.round(pts_norm[:, 0] * w), 0, width - 1).astype(np.int32)
    out[:, 1] = np.clip(np.round(pts_norm[:, 1] * h), 0, height - 1).astype(np.int32)
    return out


def draw_legend(
    canvas: np.ndarray,
    margin: int = 8,
    row_h: int = 22,
    pad: int = 6,
) -> None:
    """Draw a compact class legend in the top-left corner."""
    h, w = canvas.shape[:2]
    n = len(CLASS_NAMES)
    box_w = 200
    box_h = margin * 2 + n * row_h
    x1, y1 = margin, margin
    x2 = min(x1 + box_w, w - 1)
    y2 = min(y1 + box_h, h - 1)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)

    for i, name in enumerate(CLASS_NAMES):
        cy = y1 + margin + i * row_h + row_h // 2
        color = CLASS_COLORS_BGR[i] if i < len(CLASS_COLORS_BGR) else (200, 200, 200)
        sx = x1 + pad
        cv2.rectangle(canvas, (sx, cy - 8), (sx + 18, cy + 8), color, -1)
        cv2.putText(
            canvas,
            f"{i}: {name}",
            (sx + 24, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def overlay_labels(
    bgr: np.ndarray,
    label_path: Path,
    *,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Parse label file and draw all polygons. Unknown class IDs use gray.
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()
    text = label_path.read_text(encoding="utf-8")
    unknown = 0
    for raw_line in text.splitlines():
        parsed = parse_yolo_seg_line(raw_line)
        if parsed is None:
            continue
        cid, pts_n = parsed
        pts_px = denormalize_polygon(pts_n, w, h)
        if cid < 0 or cid >= len(CLASS_COLORS_BGR):
            color = (180, 180, 180)
            unknown += 1
        else:
            color = CLASS_COLORS_BGR[cid]
        poly = pts_px.reshape(-1, 1, 2)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
        # Small class tag at first vertex
        vx, vy = int(pts_px[0, 0]), int(pts_px[0, 1])
        cv2.putText(
            out,
            str(cid),
            (vx + 2, max(vy - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    draw_legend(out)
    if unknown:
        m = 8
        cv2.putText(
            out,
            f"unknown_cls={unknown}",
            (m, h - m),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO seg overlay sanity check")
    parser.add_argument("--images-dir", type=Path, default=None, help="default: <root>/data/processed/segmentation/yolo_exp1_base/images/train")
    parser.add_argument("--labels-dir", type=Path, default=None, help="default: <root>/data/processed/segmentation/yolo_exp1_base/labels/train")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <root>/artifacts/sanity_checks")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--line-thickness", type=int, default=2)
    args = parser.parse_args()

    root = project_root_from_script()
    images_dir = args.images_dir or (root / "data" / "processed" / "segmentation" / "yolo_exp1_base" / "images" / "train")
    labels_dir = args.labels_dir or (root / "data" / "processed" / "segmentation" / "yolo_exp1_base" / "labels" / "train")
    out_dir = args.out_dir or (root / "artifacts" / "sanity_checks")

    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1
    if not labels_dir.is_dir():
        print(f"Labels directory not found: {labels_dir}", file=sys.stderr)
        return 1

    pngs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".png")
    if not pngs:
        print(f"No PNG files in {images_dir}", file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)
    n = min(int(args.n_samples), len(pngs))
    chosen = random.sample(pngs, n)

    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in chosen:
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.is_file():
            print(f"Skip (no label): {img_path.name} -> expected {lbl_path.name}", file=sys.stderr)
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Failed to read image: {img_path}", file=sys.stderr)
            continue

        vis = overlay_labels(bgr, lbl_path, line_thickness=int(args.line_thickness))
        out_path = out_dir / f"sanity_{stem}.png"
        if not cv2.imwrite(str(out_path), vis):
            print(f"Failed to write: {out_path}", file=sys.stderr)
            continue
        print(f"Wrote {out_path.relative_to(root)}")

    print(f"Done. {n} samples requested; outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
