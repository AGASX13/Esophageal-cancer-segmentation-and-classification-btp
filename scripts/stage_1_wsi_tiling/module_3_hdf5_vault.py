"""Module 3: HDF5 coordinate vault builder for whole slide images.

This module saves valid Level 0 patch coordinates into a compact HDF5 file with
the tiling hyperparameters stored as dataset attributes for reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from module_1_tissue_masker import generate_tissue_mask
from module_2_grid_generator import generate_patch_coordinates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
COORDINATE_VAULT_DIR = PROCESSED_DIR / "coordinate_vaults"


def _vault_stem_from_wsi_path(wsi_path: str | Path) -> str:
    """Extract a stable output stem from a WSI path."""
    path = Path(wsi_path)
    stem = path.name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return stem


def coordinate_vault_path(wsi_path: str | Path) -> Path:
    """Return the HDF5 vault path for a WSI without creating the file."""
    return COORDINATE_VAULT_DIR / f"{_vault_stem_from_wsi_path(wsi_path)}.h5"


def save_coordinates_to_h5(
    wsi_path,
    coordinates,
    patch_size,
    downsample_factor,
    tissue_threshold,
):
    """Save patch coordinates and tiling metadata to an HDF5 vault.

    Args:
        wsi_path: Path to the source WSI.
        coordinates: Iterable of ``(x, y)`` Level 0 coordinate pairs.
        patch_size: Level 0 patch size used by the grid generator.
        downsample_factor: Downsample factor used to map Level 0 to mask space.
        tissue_threshold: Minimum tissue ratio used to keep coordinates.

    Returns:
        Path to the saved ``.h5`` file.
    """
    COORDINATE_VAULT_DIR.mkdir(parents=True, exist_ok=True)

    coords_array = np.asarray(list(coordinates), dtype=np.int32)
    if coords_array.size == 0:
        coords_array = coords_array.reshape(0, 2)
    elif coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(
            "coordinates must be an iterable of (x, y) pairs; "
            f"got array shape {coords_array.shape}"
        )

    output_path = coordinate_vault_path(wsi_path)

    with h5py.File(output_path, "w") as h5_file:
        coords_dataset = h5_file.create_dataset(
            "coords",
            data=coords_array,
            dtype=np.int32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        coords_dataset.attrs["patch_size"] = int(patch_size)
        coords_dataset.attrs["downsample_factor"] = float(downsample_factor)
        coords_dataset.attrs["tissue_threshold"] = float(tissue_threshold)

    return output_path


def process_single_wsi(
    wsi_path,
    patch_size=256,
    tissue_threshold=0.5,
):
    """Run Modules 1-3 for one WSI and return the saved vault path."""
    mask, downsample_factor = generate_tissue_mask(wsi_path)
    coordinates = generate_patch_coordinates(
        wsi_path,
        mask,
        downsample_factor,
        patch_size=patch_size,
        tissue_threshold=tissue_threshold,
    )

    return save_coordinates_to_h5(
        wsi_path,
        coordinates,
        patch_size,
        downsample_factor,
        tissue_threshold,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HDF5 coordinate vault from a single WSI.",
    )
    parser.add_argument("wsi_path", type=Path, help="Path to a .svs WSI file.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Level 0 patch size in pixels.",
    )
    parser.add_argument(
        "--tissue-threshold",
        type=float,
        default=0.5,
        help="Minimum tissue fraction required to keep a patch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("Generating tissue mask...")
    output_path = process_single_wsi(
        args.wsi_path,
        patch_size=args.patch_size,
        tissue_threshold=args.tissue_threshold,
    )

    print(f"Saved coordinate vault to {output_path}")


if __name__ == "__main__":
    main()
