"""Module 2: patch grid generation for whole slide images.

This module converts a low-resolution tissue mask from Module 1 into a list of
Level 0 patch coordinates that contain enough tissue for extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide

from module_1_tissue_masker import generate_tissue_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
DEFAULT_GRID_DEBUG_PATH = INTERIM_DIR / "grid_debug.png"


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 2D mask array suitable for tissue-ratio calculations."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {mask_array.shape}")
    if mask_array.size == 0:
        raise ValueError("mask must not be empty")
    return mask_array


def _mask_window_for_patch(
    x: int,
    y: int,
    level0_width: int,
    level0_height: int,
    mask_width: int,
    mask_height: int,
    downsample_factor: float,
    patch_size: int,
) -> tuple[int, int, int, int]:
    """Map a Level 0 patch window to a clamped mask-space window."""
    patch_x_end = min(x + patch_size, level0_width)
    patch_y_end = min(y + patch_size, level0_height)

    mask_x0 = int(np.floor(x / downsample_factor))
    mask_y0 = int(np.floor(y / downsample_factor))
    mask_x1 = int(np.ceil(patch_x_end / downsample_factor))
    mask_y1 = int(np.ceil(patch_y_end / downsample_factor))

    mask_x0 = max(0, min(mask_x0, mask_width))
    mask_y0 = max(0, min(mask_y0, mask_height))
    mask_x1 = max(mask_x0, min(mask_x1, mask_width))
    mask_y1 = max(mask_y0, min(mask_y1, mask_height))

    return mask_x0, mask_y0, mask_x1, mask_y1


def generate_patch_coordinates(
    wsi_path,
    mask,
    downsample_factor,
    patch_size=256,
    tissue_threshold=0.5,
):
    """Generate valid Level 0 patch coordinates for a WSI.

    Args:
        wsi_path: Path to the WSI file.
        mask: Binary tissue mask generated at a lower WSI pyramid level.
        downsample_factor: Downsample factor between Level 0 and mask space.
        patch_size: Level 0 patch size in pixels.
        tissue_threshold: Minimum tissue fraction required to keep a patch.

    Returns:
        A list of ``(x, y)`` Level 0 coordinate tuples.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than 0")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be greater than 0")
    if not 0 <= tissue_threshold <= 1:
        raise ValueError("tissue_threshold must be between 0 and 1")

    mask_array = _validate_mask(mask)
    mask_height, mask_width = mask_array.shape
    valid_coordinates: list[tuple[int, int]] = []

    with openslide.OpenSlide(str(wsi_path)) as slide:
        level0_width, level0_height = slide.dimensions

    for y in range(0, level0_height, patch_size):
        for x in range(0, level0_width, patch_size):
            mask_x0, mask_y0, mask_x1, mask_y1 = _mask_window_for_patch(
                x=x,
                y=y,
                level0_width=level0_width,
                level0_height=level0_height,
                mask_width=mask_width,
                mask_height=mask_height,
                downsample_factor=float(downsample_factor),
                patch_size=patch_size,
            )

            mask_region = mask_array[mask_y0:mask_y1, mask_x0:mask_x1]
            if mask_region.size == 0:
                continue

            tissue_ratio = np.count_nonzero(mask_region) / mask_region.size
            if tissue_ratio >= tissue_threshold:
                valid_coordinates.append((x, y))

    return valid_coordinates


def visualize_grid(mask, coordinates, downsample_factor, output_path=DEFAULT_GRID_DEBUG_PATH):
    """Save a grid-debug image with valid Level 0 patch coordinates over the mask."""
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be greater than 0")

    mask_array = _validate_mask(mask)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if coordinates:
        coords = np.asarray(coordinates, dtype=float)
        plot_x = coords[:, 0] / float(downsample_factor)
        plot_y = coords[:, 1] / float(downsample_factor)
    else:
        plot_x = np.array([])
        plot_y = np.array([])

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(mask_array, cmap="gray", vmin=0, vmax=max(1, int(mask_array.max())))
    ax.scatter(plot_x, plot_y, s=8, c="red", marker=".", alpha=0.8)
    ax.set_title(f"Valid patch grid ({len(coordinates)} patches)")
    ax.set_xlim(0, mask_array.shape[1])
    ax.set_ylim(mask_array.shape[0], 0)
    ax.axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Level 0 patch coordinates from a WSI tissue mask.",
    )
    parser.add_argument("wsi_path", type=Path, help="Path to the WSI file.")
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
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=DEFAULT_GRID_DEBUG_PATH,
        help="Path for the grid debug image.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mask, downsample_factor = generate_tissue_mask(args.wsi_path)
    coordinates = generate_patch_coordinates(
        args.wsi_path,
        mask,
        downsample_factor,
        patch_size=args.patch_size,
        tissue_threshold=args.tissue_threshold,
    )
    debug_path = visualize_grid(
        mask,
        coordinates,
        downsample_factor,
        args.debug_output,
    )
    print(f"Generated {len(coordinates)} valid patch coordinates")
    print(f"Saved grid debug plot to {debug_path}")


if __name__ == "__main__":
    main()
