"""Module 4: fault-tolerant sequential Stage 3 embedding orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from module_3_extractor import DEFAULT_OUTPUT_DIR, extract_slide_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT_DIR = PROJECT_ROOT / "data" / "interim" / "yolo_filtered_vaults"
DEFAULT_WSI_ROOT = PROJECT_ROOT / "data" / "raw" / "tcga_esca_wsi"


def _base_name(path: Path) -> str:
    stem = path.name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return stem


def _resolve_wsi_for_vault(vault_path: Path, wsi_root: Path) -> Path:
    base_name = _base_name(vault_path)
    matches = list(wsi_root.rglob(f"{base_name}*.svs"))
    if not matches:
        raise FileNotFoundError(f"No WSI match found for vault: {vault_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple WSI matches found for {vault_path}: {matches}")
    return matches[0]


def run_stage3_batch(
    vault_dir: Path = DEFAULT_VAULT_DIR,
    wsi_root: Path = DEFAULT_WSI_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, int]:
    """Process all pending YOLO-filtered vaults sequentially and safely."""
    vault_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(vault_dir.glob("*.h5"))
    pending_h5 = [h5_path for h5_path in h5_files if not (output_dir / f"{_base_name(h5_path)}.pt").exists()]

    processed = 0
    skipped = len(h5_files) - len(pending_h5)
    failed = 0

    print(f"Discovered {len(h5_files)} filtered vaults in {vault_dir}")
    print(f"Skipping {skipped} slides with existing feature tensors")
    print(f"Processing {len(pending_h5)} slides sequentially")

    for h5_path in tqdm(pending_h5, desc="Stage 3 embedding", unit="slide"):
        try:
            wsi_path = _resolve_wsi_for_vault(h5_path, wsi_root)
            extract_slide_embeddings(
                wsi_path=wsi_path,
                h5_path=h5_path,
                output_dir=output_dir,
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] Failed {h5_path.name}: {type(exc).__name__}: {exc}")

    return {
        "discovered": len(h5_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 3 embedding extraction across all filtered vaults.",
    )
    parser.add_argument(
        "--vault-dir",
        type=Path,
        default=DEFAULT_VAULT_DIR,
        help=f"Directory containing Stage 2 filtered .h5 vaults (default: {DEFAULT_VAULT_DIR}).",
    )
    parser.add_argument(
        "--wsi-root",
        type=Path,
        default=DEFAULT_WSI_ROOT,
        help=f"Root directory containing .svs WSIs (default: {DEFAULT_WSI_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output feature tensors (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_stage3_batch(
        vault_dir=args.vault_dir,
        wsi_root=args.wsi_root,
        output_dir=args.output_dir,
    )
    print(
        f"Stage 3 complete | processed={summary['processed']} | "
        f"skipped={summary['skipped']} | failed={summary['failed']}"
    )


if __name__ == "__main__":
    main()
