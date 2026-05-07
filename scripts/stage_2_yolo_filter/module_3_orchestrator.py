"""Module 3: sequential, fault-tolerant Stage 2 YOLO orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from module_1_yolo_inference import filter_coordinate_vault_with_yolo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_VAULT_DIR = PROJECT_ROOT / "data" / "processed" / "coordinate_vaults"
STAGE2_FILTERED_DIR = PROJECT_ROOT / "data" / "interim" / "yolo_filtered_vaults"
RAW_WSI_ROOT = PROJECT_ROOT / "data" / "raw" / "tcga_esca_wsi"


def _discover_stage1_vaults(vault_dir: Path) -> list[Path]:
    return sorted(vault_dir.glob("*.h5"))


def _discover_finished_filtered(filtered_dir: Path) -> set[str]:
    return {path.stem for path in filtered_dir.glob("*.h5")}


def _find_matching_wsi(vault_path: Path, wsi_root: Path) -> Path:
    stem = vault_path.stem
    matches: list[Path] = []
    for class_dir in ("cancerous", "non_cancerous"):
        candidate_dir = wsi_root / class_dir
        if candidate_dir.is_dir():
            matches.extend(candidate_dir.rglob(f"{stem}*.svs"))

    if not matches:
        raise FileNotFoundError(f"No matching WSI found for vault {vault_path.name}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple WSI matches found for {vault_path.name}: {matches}")
    return matches[0]


def run_stage2_batch(
    stage1_vault_dir: Path = STAGE1_VAULT_DIR,
    stage2_filtered_dir: Path = STAGE2_FILTERED_DIR,
    raw_wsi_root: Path = RAW_WSI_ROOT,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, int]:
    """Process all pending Stage 1 vaults through Stage 2 YOLO filtering."""
    stage2_filtered_dir.mkdir(parents=True, exist_ok=True)

    stage1_vaults = _discover_stage1_vaults(stage1_vault_dir)
    finished_stems = _discover_finished_filtered(stage2_filtered_dir)
    pending_vaults = [
        vault_path for vault_path in stage1_vaults if vault_path.stem not in finished_stems
    ]

    processed = 0
    skipped = len(stage1_vaults) - len(pending_vaults)
    failed = 0

    print(f"Discovered {len(stage1_vaults)} Stage 1 vaults in {stage1_vault_dir}")
    print(f"Skipping {skipped} slides with existing filtered vaults")
    print(f"Processing {len(pending_vaults)} pending slides sequentially")

    for stage1_h5 in tqdm(pending_vaults, desc="Stage 2 YOLO", unit="slide"):
        try:
            wsi_path = _find_matching_wsi(stage1_h5, raw_wsi_root)
            filter_coordinate_vault_with_yolo(
                wsi_path=wsi_path,
                stage1_vault_path=stage1_h5,
                output_dir=stage2_filtered_dir,
                yolo_weights_path=(
                    PROJECT_ROOT
                    / "models"
                    / "segmentation"
                    / "experiment_2"
                    / "yolov8m_pannuke_esophagus_exp2_augmented_best.pt"
                ),
                batch_size=batch_size,
                num_workers=num_workers,
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] Failed {stage1_h5.name}: {type(exc).__name__}: {exc}")

    return {"processed": processed, "skipped": skipped, "failed": failed}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 YOLO filtering sequentially for all Stage 1 vaults.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="YOLO inference batch size (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for patch extraction (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_stage2_batch(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Successfully processed {summary['processed']} slides, "
        f"skipped {summary['skipped']}, failed {summary['failed']}"
    )


if __name__ == "__main__":
    main()
