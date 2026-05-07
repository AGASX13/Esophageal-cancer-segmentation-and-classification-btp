#!/usr/bin/env python3
"""
Experiment 2 Phase 4: train YOLOv8 Medium segmentation on the augmented dataset.

Uses data_exp2.yaml, which maps data/processed/segmentation/yolo_exp2_refined_augmented train/val splits.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


PRETRAINED_MODEL = Path("models/segmentation/pretrained/yolov8m_seg_coco_pretrained.pt")


def project_root_from_script() -> Path:
    """Assume this file lives in <root>/scripts/segmentation_experiments/experiment_02_augmented_yolov8m/."""
    return Path(__file__).resolve().parents[3]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> int:
    setup_logging()

    root = project_root_from_script()
    os.chdir(root)

    data_yaml = root / "data_exp2.yaml"
    if not data_yaml.is_file():
        logging.error("Missing Experiment 2 data config: %s", data_yaml)
        return 1

    device = 0 if torch.cuda.is_available() else "cpu"
    logging.info("Experiment 2 data config: %s", data_yaml.resolve())
    logging.info("Training device: %s", device)

    # [EXP 2: MODEL UPSCALING] Experiment 2 uses the heavier YOLOv8 Medium segmentation
    # model so it can learn from elastic deformations and hard negatives; batch=16 is
    # intentionally conservative to reduce VRAM pressure compared with smaller models.
    pretrained_model = root / PRETRAINED_MODEL
    model = YOLO(str(pretrained_model))
    model.train(
        data="data_exp2.yaml",
        epochs=100,
        imgsz=256,
        batch=16,
        project="runs/segment",
        name="exp2_medium_augmented",
        device=device,
    )

    logging.info("Experiment 2 YOLOv8m segmentation training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
