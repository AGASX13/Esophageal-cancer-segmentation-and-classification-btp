#!/usr/bin/env python3
"""
Experiment 2 centralized evaluation engine.

Runs YOLOv8 validation for the Experiment 2 Medium segmentation model, routes all
Ultralytics plots into artifacts/exp2_evaluation, and writes a readable metrics
summary with box, mask, F1, and per-class metrics.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


MODEL_WEIGHTS = Path(
    "models/segmentation/experiment_2/"
    "yolov8m_pannuke_esophagus_exp2_augmented_best.pt"
)
DATA_YAML = Path("data_exp2.yaml")
ARTIFACT_PROJECT = Path("artifacts")
EVALUATION_NAME = "exp2_evaluation"
SUMMARY_FILENAME = "EXP2_METRICS_SUMMARY.txt"
CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


@dataclass(frozen=True)
class MetricRow:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


def project_root_from_script() -> Path:
    """Assume this file lives in <root>/scripts/segmentation_experiments/experiment_02_augmented_yolov8m/."""
    return Path(__file__).resolve().parents[3]


def configure_logging(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "evaluation.log"

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    logging.info("[EXP 2: EVALUATION] Evaluation log routed to %s", log_path.resolve())


def safe_f1(precision: float, recall: float) -> float:
    # [EXP 2: EVALUATION] F1 is the harmonic mean of precision and recall; guard
    # against zero-denominator cases so empty predictions/classes report 0.0 safely.
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return 2.0 * (precision * recall) / denom


def metric_value(metric_obj: Any, attr_name: str) -> float:
    value = getattr(metric_obj, attr_name, 0.0)
    try:
        return float(value)
    except TypeError:
        return 0.0


def extract_overall_metrics(metric_obj: Any) -> MetricRow:
    # [EXP 2: EVALUATION] Ultralytics stores overall box/mask metrics on metrics.box
    # and metrics.seg; extract precision, recall, mAP50, and mAP50-95 consistently.
    precision = metric_value(metric_obj, "mp")
    recall = metric_value(metric_obj, "mr")
    return MetricRow(
        precision=precision,
        recall=recall,
        f1=safe_f1(precision, recall),
        map50=metric_value(metric_obj, "map50"),
        map50_95=metric_value(metric_obj, "map"),
    )


def extract_class_metrics(metric_obj: Any, class_index: int) -> MetricRow:
    class_result = getattr(metric_obj, "class_result", None)
    if not callable(class_result):
        return MetricRow(precision=0.0, recall=0.0, f1=0.0, map50=0.0, map50_95=0.0)

    try:
        precision, recall, map50, map50_95 = class_result(class_index)
    except (IndexError, TypeError, ValueError):
        return MetricRow(precision=0.0, recall=0.0, f1=0.0, map50=0.0, map50_95=0.0)

    precision = float(precision)
    recall = float(recall)
    return MetricRow(
        precision=precision,
        recall=recall,
        f1=safe_f1(precision, recall),
        map50=float(map50),
        map50_95=float(map50_95),
    )


def load_class_names(data_yaml: Path) -> list[str]:
    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        logging.warning("Could not parse class names from %s; using defaults", data_yaml)
        return CLASS_NAMES

    names = payload.get("names", CLASS_NAMES)
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return CLASS_NAMES


def format_metric_row(label: str, row: MetricRow) -> str:
    return (
        f"{label:<18} | "
        f"{row.precision:>9.4f} | "
        f"{row.recall:>9.4f} | "
        f"{row.f1:>9.4f} | "
        f"{row.map50:>9.4f} | "
        f"{row.map50_95:>9.4f}"
    )


def write_metric_table(lines: list[str], title: str, rows: list[tuple[str, MetricRow]]) -> None:
    lines.extend(
        [
            title,
            "-" * len(title),
            f"{'Metric Scope':<18} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | {'mAP50':>9} | {'mAP50-95':>9}",
            "-" * 82,
        ]
    )
    for label, row in rows:
        lines.append(format_metric_row(label, row))
    lines.append("")


def write_summary_report(metrics: Any, summary_path: Path, weights_path: Path, data_yaml: Path) -> None:
    class_names = load_class_names(data_yaml)
    box_overall = extract_overall_metrics(metrics.box)
    mask_overall = extract_overall_metrics(metrics.seg)

    # [EXP 2: EVALUATION] Per-class values are extracted separately for bounding
    # boxes and segmentation masks so the report can show class-specific behavior.
    box_rows = [("Overall", box_overall)]
    mask_rows = [("Overall", mask_overall)]
    for idx, class_name in enumerate(class_names):
        box_rows.append((class_name, extract_class_metrics(metrics.box, idx)))
        mask_rows.append((class_name, extract_class_metrics(metrics.seg, idx)))

    lines: list[str] = [
        "Experiment 2 YOLOv8m Segmentation Evaluation Summary",
        "=" * 54,
        "",
        f"Model weights: {weights_path.resolve()}",
        f"Dataset config: {data_yaml.resolve()}",
        f"Ultralytics artifact directory: {Path(metrics.save_dir).resolve()}",
        "",
    ]
    write_metric_table(lines, "Bounding Box Metrics", box_rows)
    write_metric_table(lines, "Segmentation Mask Metrics", mask_rows)

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("[EXP 2: EVALUATION] Summary report written to %s", summary_path.resolve())


def main() -> int:
    root = project_root_from_script()
    os.chdir(root)

    artifact_dir = root / ARTIFACT_PROJECT / EVALUATION_NAME
    configure_logging(artifact_dir)

    weights_path = root / MODEL_WEIGHTS
    data_yaml = root / DATA_YAML
    if not weights_path.is_file():
        logging.error("Model weights not found: %s", weights_path)
        return 1
    if not data_yaml.is_file():
        logging.error("Dataset YAML not found: %s", data_yaml)
        return 1

    logging.info("[EXP 2: EVALUATION] Loading model: %s", weights_path.resolve())
    model = YOLO(str(weights_path))

    logging.info(
        "[EXP 2: EVALUATION] Running validation with plots routed to artifacts/%s",
        EVALUATION_NAME,
    )
    # [EXP 2: EVALUATION] Force validation plots, curves, and run metadata into
    # artifacts/exp2_evaluation instead of the training runs/ directory.
    metrics = model.val(
        data="data_exp2.yaml",
        split="val",
        plots=True,
        project="artifacts",
        name="exp2_evaluation",
        exist_ok=True,
        workers=0,
    )

    summary_path = artifact_dir / SUMMARY_FILENAME
    write_summary_report(
        metrics=metrics,
        summary_path=summary_path,
        weights_path=weights_path,
        data_yaml=data_yaml,
    )

    logging.info("[EXP 2: EVALUATION] Evaluation complete. Artifacts saved to %s", artifact_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
