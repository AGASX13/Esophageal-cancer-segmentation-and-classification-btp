"""Module 4: multiprocessing orchestrator for Stage 1 WSI tiling.

This script scans the TCGA ESCA WSI dataset, skips slides that already have
coordinate vaults, and processes the remaining slides in parallel.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from module_3_hdf5_vault import (
    COORDINATE_VAULT_DIR,
    coordinate_vault_path,
    process_single_wsi,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_WSI_DIR = DATA_DIR / "raw" / "tcga_esca_wsi"
DEFAULT_PATCH_SIZE = 256
DEFAULT_TISSUE_THRESHOLD = 0.5


def discover_wsi_files(wsi_root: Path = RAW_WSI_DIR) -> list[Path]:
    """Find all .svs files under cancerous and non_cancerous WSI folders."""
    wsi_files: list[Path] = []
    for class_dir_name in ("cancerous", "non_cancerous"):
        class_dir = wsi_root / class_dir_name
        if class_dir.exists():
            wsi_files.extend(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".svs"
            )
    return sorted(wsi_files)


def split_pending_and_skipped(wsi_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Separate slides needing processing from slides with existing vaults."""
    COORDINATE_VAULT_DIR.mkdir(parents=True, exist_ok=True)

    pending: list[Path] = []
    skipped: list[Path] = []
    for wsi_path in wsi_files:
        if coordinate_vault_path(wsi_path).exists():
            skipped.append(wsi_path)
        else:
            pending.append(wsi_path)
    return pending, skipped


def _process_wsi_safely(
    wsi_path: Path,
    patch_size: int,
    tissue_threshold: float,
) -> tuple[str, str, str]:
    """Process one WSI and convert failures into structured results."""
    try:
        output_path = process_single_wsi(
            wsi_path,
            patch_size=patch_size,
            tissue_threshold=tissue_threshold,
        )
        return "success", str(wsi_path), str(output_path)
    except Exception as exc:
        return "failed", str(wsi_path), f"{type(exc).__name__}: {exc}"


def run_batch(
    wsi_root: Path = RAW_WSI_DIR,
    patch_size: int = DEFAULT_PATCH_SIZE,
    tissue_threshold: float = DEFAULT_TISSUE_THRESHOLD,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Run Stage 1 vault generation across the WSI dataset."""
    wsi_files = discover_wsi_files(wsi_root)
    pending, skipped = split_pending_and_skipped(wsi_files)

    worker_count = max_workers
    if worker_count is None:
        worker_count = max(1, (os.cpu_count() or 1) - 2)
    worker_count = max(1, worker_count)

    successful = 0
    failed = 0

    print(f"Found {len(wsi_files)} .svs files under {wsi_root}")
    print(f"Skipping {len(skipped)} slides with existing HDF5 vaults")
    print(f"Processing {len(pending)} slides with {worker_count} workers")

    if pending:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_wsi_safely,
                    wsi_path,
                    patch_size,
                    tissue_threshold,
                )
                for wsi_path in pending
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building coordinate vaults",
            ):
                status, wsi_path, message = future.result()
                if status == "success":
                    successful += 1
                else:
                    failed += 1
                    print(f"[ERROR] Failed {wsi_path}: {message}")

    return {
        "processed": successful,
        "skipped": len(skipped),
        "failed": failed,
        "discovered": len(wsi_files),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 WSI masking, grid generation, and HDF5 vaulting.",
    )
    parser.add_argument(
        "--wsi-root",
        type=Path,
        default=RAW_WSI_DIR,
        help="Root directory containing cancerous/ and non_cancerous/ WSI folders.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Level 0 patch size in pixels.",
    )
    parser.add_argument(
        "--tissue-threshold",
        type=float,
        default=DEFAULT_TISSUE_THRESHOLD,
        help="Minimum tissue fraction required to keep a patch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to os.cpu_count() - 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_batch(
        wsi_root=args.wsi_root,
        patch_size=args.patch_size,
        tissue_threshold=args.tissue_threshold,
        max_workers=args.workers,
    )
    print(
        "Successfully processed "
        f"{summary['processed']} slides, skipped {summary['skipped']}, "
        f"failed {summary['failed']}"
    )


if __name__ == "__main__":
    main()
