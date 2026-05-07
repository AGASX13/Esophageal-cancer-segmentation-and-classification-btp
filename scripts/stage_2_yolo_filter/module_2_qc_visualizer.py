"""Stage 2 QC visualizer for YOLO-filtered coordinate vaults.

This script overlays surviving Stage 2 coordinates onto a Stage 1 tissue mask
to produce a quick quality-control map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "yolo_debug_plots"

# Import Stage 1 tissue masker from sibling script directory.
STAGE1_DIR = Path(__file__).resolve().parent.parent / "stage_1_wsi_tiling"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))
from module_1_tissue_masker import generate_tissue_mask  # noqa: E402


def _wsi_base_name(wsi_path: Path) -> str:
    """Return stable slide base name without chained suffixes."""
    stem = wsi_path.name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return stem


def _load_filtered_coords(vault_path: Path) -> np.ndarray:
    """Load filtered coords from an HDF5 vault."""
    with h5py.File(vault_path, "r") as h5_file:
        if "coords" not in h5_file:
            raise KeyError(f"Dataset 'coords' not found in {vault_path}")
        coords = h5_file["coords"][:]

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords shape (N, 2), got {coords.shape}")

    return coords


def create_yolo_qc_plot(
    wsi_path: Path,
    filtered_vault_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    point_size: int = 4,
    point_alpha: float = 0.6,
    point_color: str = "lime",
) -> Path:
    """Create and save a QC overlay of YOLO-surviving coordinates."""
    if not filtered_vault_path.is_file():
        raise FileNotFoundError(f"Filtered vault not found: {filtered_vault_path}")

    tissue_mask, downsample_factor = generate_tissue_mask(wsi_path)
    coords = _load_filtered_coords(filtered_vault_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_wsi_base_name(wsi_path)}_yolo_qc.png"

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(tissue_mask, cmap="gray", vmin=0, vmax=1)

    if len(coords) > 0:
        scaled_coords = coords.astype(np.float32) / float(downsample_factor)
        ax.scatter(
            scaled_coords[:, 0],
            scaled_coords[:, 1],
            c=point_color,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
        )

    ax.set_title(f"YOLO surviving patches: {len(coords)}")
    ax.axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Stage 2 QC visualization by overlaying YOLO-filtered "
            "coordinates on the Stage 1 tissue mask."
        )
    )
    parser.add_argument("wsi_path", type=Path, help="Path to input WSI (.svs).")
    parser.add_argument(
        "filtered_vault_path",
        type=Path,
        help="Path to Stage 2 YOLO-filtered coordinate vault (.h5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save QC plot (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=4,
        help="Scatter marker size for surviving coordinates.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.6,
        help="Scatter marker alpha for surviving coordinates.",
    )
    parser.add_argument(
        "--point-color",
        type=str,
        default="lime",
        help="Scatter marker color for surviving coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = create_yolo_qc_plot(
        wsi_path=args.wsi_path,
        filtered_vault_path=args.filtered_vault_path,
        output_dir=args.output_dir,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        point_color=args.point_color,
    )
    print(f"Saved YOLO QC plot to {output_path}")


if __name__ == "__main__":
    main()
