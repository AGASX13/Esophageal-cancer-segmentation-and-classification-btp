"""Experiment 1 orchestrator: baseline YOLOv8s segmentation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineStep:
    number: int
    name: str
    script_name: str


STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(1, "Prepare YOLO dataset from PanNuke", "module_01_prepare_yolo_dataset.py"),
    PipelineStep(2, "Validate YOLO label overlays", "module_02_validate_label_overlays.py"),
    PipelineStep(3, "Augment YOLO training split", "module_03_augment_training_split.py"),
    PipelineStep(4, "Train YOLOv8s segmentation model", "module_04_train_yolov8s_segmentation.py"),
    PipelineStep(5, "Evaluate YOLOv8s segmentation model", "module_05_evaluate_yolov8s_segmentation.py"),
    PipelineStep(6, "Plot YOLOv8 training curves", "module_06_plot_training_curves.py"),
)


def selected_steps(start_at: int, stop_after: int) -> tuple[PipelineStep, ...]:
    return tuple(step for step in STEPS if start_at <= step.number <= stop_after)


def run_step(step: PipelineStep, dry_run: bool) -> None:
    script_path = EXPERIMENT_DIR / step.script_name
    command = [sys.executable, str(script_path)]

    print(f"\n[EXP1:{step.number:02d}] {step.name}")
    print(" ".join(command))
    if dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 baseline YOLOv8s segmentation modules in order.",
    )
    parser.add_argument("--start-at", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--stop-after", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.start_at > args.stop_after:
        raise ValueError("--start-at must be less than or equal to --stop-after")

    for step in selected_steps(args.start_at, args.stop_after):
        run_step(step, dry_run=args.dry_run)

    print("\nExperiment 1 pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
