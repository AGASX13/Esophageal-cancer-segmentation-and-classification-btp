from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path


DEFAULT_WEIGHTS = Path(
    "models/segmentation/experiment_1/"
    "yolov8s_pannuke_esophagus_exp1_final_best.pt"
)
DEFAULT_DATA = Path("data.yaml")
DEFAULT_PROJECT = Path("runs/segment")
DEFAULT_NAME = "esophagus_val"
DEFAULT_REPORT_DIR = Path("artifacts/evaluations/final_report")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a trained YOLOv8 segmentation model and package key artifacts."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained YOLOv8 weights.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Ultralytics project directory for validation outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Run name for Ultralytics validation outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory to collect final presentation-ready artifacts.",
    )
    return parser


def run_validation(weights_path: Path, data_path: Path, project_dir: Path, run_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is not installed. Install it before running evaluation.") from exc

    logging.info("Loading model weights from %s", weights_path)
    model = YOLO(str(weights_path))

    logging.info("Running validation on split='val' with workers=0 for Windows compatibility")
    results = model.val(
        data=str(data_path),
        split="val",
        project=str(project_dir),
        name=run_name,
        workers=0,
        exist_ok=True,
    )
    logging.info("Validation completed. Results saved to %s", results.save_dir)
    return results


def write_summary(results, output_path: Path, weights_path: Path, data_path: Path) -> None:
    metrics = results.results_dict
    summary = "\n".join(
        [
            "YOLOv8 Instance Segmentation Evaluation Summary",
            "=" * 46,
            "",
            f"Model weights: {weights_path}",
            f"Dataset config: {data_path}",
            f"Validation output directory: {results.save_dir}",
            "",
            "Bounding Box Metrics",
            f"  mAP50:    {metrics['metrics/mAP50(B)']:.4f}",
            f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}",
            "",
            "Segmentation Mask Metrics",
            f"  mAP50:    {metrics['metrics/mAP50(M)']:.4f}",
            f"  mAP50-95: {metrics['metrics/mAP50-95(M)']:.4f}",
            "",
        ]
    )
    output_path.write_text(summary, encoding="utf-8")
    logging.info("Saved evaluation summary to %s", output_path)


def copy_if_exists(source: Path, destination_dir: Path) -> bool:
    if not source.exists():
        return False
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    logging.info("Copied %s -> %s", source, destination)
    return True


def harvest_artifacts(run_dir: Path, report_dir: Path) -> None:
    preferred_confusion = run_dir / "confusion_matrix.png"
    normalized_confusion = run_dir / "confusion_matrix_normalized.png"

    if not copy_if_exists(preferred_confusion, report_dir):
        if not copy_if_exists(normalized_confusion, report_dir):
            logging.warning("No confusion matrix image found in %s", run_dir)

    for fixed_name in ["MaskPR_curve.png", "MaskF1_curve.png"]:
        if not copy_if_exists(run_dir / fixed_name, report_dir):
            logging.warning("Expected artifact not found: %s", run_dir / fixed_name)

    prediction_files = sorted(run_dir.glob("val_batch*_pred.jpg"))
    if not prediction_files:
        logging.warning("No validation prediction preview images found in %s", run_dir)
        return

    for prediction_file in prediction_files:
        copy_if_exists(prediction_file, report_dir)


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    root = project_root()
    weights_path = (args.weights if args.weights.is_absolute() else root / args.weights).resolve()
    data_path = (args.data if args.data.is_absolute() else root / args.data).resolve()
    project_dir = (args.project if args.project.is_absolute() else root / args.project).resolve()
    report_dir = (args.report_dir if args.report_dir.is_absolute() else root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    results = run_validation(weights_path, data_path, project_dir, args.name)

    write_summary(
        results=results,
        output_path=report_dir / "evaluation_summary.txt",
        weights_path=weights_path,
        data_path=data_path,
    )
    harvest_artifacts(Path(results.save_dir), report_dir)

    logging.info("Final report packaging completed at %s", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
