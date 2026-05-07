#!/usr/bin/env python3
"""
Train YOLOv8 instance segmentation on augmented esophagus patches (Ultralytics).

Writes data.yaml at the project root, then trains from the registered YOLOv8s
segmentation pretrained checkpoint with train/val split:
  - train: data/processed/segmentation/yolo_exp1_augmented/images/train  (+ labels/train)
  - val:   data/processed/segmentation/yolo_exp1_base/images/val (+ labels/val)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_PRETRAINED_MODEL = "models/segmentation/pretrained/yolov8s_seg_coco_pretrained.pt"


def write_data_yaml(
    root: Path,
    *,
    train_images: Path,
    val_images: Path,
) -> Path:
    """
    Ultralytics infers label paths by replacing `images` with `labels` in the train/val paths.
    """
    train_images = train_images.resolve()
    val_images = val_images.resolve()
    if not train_images.is_dir():
        raise FileNotFoundError(f"Train images dir missing: {train_images}")
    if not val_images.is_dir():
        raise FileNotFoundError(f"Val images dir missing: {val_images}")

    train_lbl = train_images.parent.parent / "labels" / train_images.name
    val_lbl = val_images.parent.parent / "labels" / val_images.name
    if not train_lbl.is_dir():
        logging.warning("Expected train labels dir not found: %s", train_lbl)
    if not val_lbl.is_dir():
        logging.warning("Expected val labels dir not found: %s", val_lbl)

    # Absolute POSIX paths avoid YAML escaping issues on Windows.
    payload = {
        "train": str(train_images.as_posix()),
        "val": str(val_images.as_posix()),
        "nc": 5,
        "names": ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"],
    }

    out = root / "data.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8-seg training (esophagus PanNuke export)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--model", type=str, default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument("--project", type=str, default="runs/segment")
    parser.add_argument("--name", type=str, default="esophagus_train")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = project_root()
    train_img = root / "data" / "processed" / "segmentation" / "yolo_exp1_augmented" / "images" / "train"
    val_img = root / "data" / "processed" / "segmentation" / "yolo_exp1_base" / "images" / "val"

    try:
        yaml_path = write_data_yaml(
            root,
            train_images=train_img,
            val_images=val_img,
        )
    except Exception as e:
        logging.exception("Failed to write data.yaml: %s", e)
        return 1

    logging.info("data.yaml created successfully at %s", yaml_path.resolve())

    try:
        from ultralytics import YOLO
    except ImportError:
        logging.error(
            "ultralytics is not installed. Install with: pip install ultralytics"
        )
        return 1

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("PyTorch device: %s", device)

    model_path = Path(args.model)
    model_path = model_path if model_path.is_absolute() else root / model_path
    logging.info("Training Started (model=%s, data=%s)", model_path, yaml_path)

    model = YOLO(str(model_path))
    proj = Path(args.project)
    project_resolved = proj if proj.is_absolute() else (root / proj).resolve()

    model.train(
        data=str(yaml_path.resolve()),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        patience=int(args.patience),
        project=str(project_resolved),
        name=args.name,
    )

    logging.info("Training Completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
