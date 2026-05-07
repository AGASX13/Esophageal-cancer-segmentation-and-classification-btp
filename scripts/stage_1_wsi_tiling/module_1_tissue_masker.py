"""Module 1: tissue masking for TCGA whole slide images.

This module reads a low-resolution WSI pyramid level and creates a binary mask
that separates tissue from bright glass/background regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
DEFAULT_DEBUG_DIR = INTERIM_DIR / "mask_debug_plots"

TARGET_THUMBNAIL_MAX_DIMENSION = 2048
MAX_THUMBNAIL_MAX_DIMENSION = 8192


def _ensure_path_is_under(path: Path, allowed_root: Path, path_label: str) -> Path:
    """Resolve and validate that a path stays inside an allowed project area."""
    resolved_path = path.expanduser().resolve()
    resolved_root = allowed_root.resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{path_label} must be inside {resolved_root}") from exc

    return resolved_path


def _select_mask_level(slide: openslide.OpenSlide) -> tuple[int, float]:
    """Select a pyramid level using robust thumbnail dimensions.

    The selected level should be close to a 2048 px maximum dimension while
    avoiding very large thumbnails that can cause memory pressure.
    """
    level_dimensions = [
        (level, max(dimensions))
        for level, dimensions in enumerate(slide.level_dimensions)
    ]
    downsamples = [float(downsample) for downsample in slide.level_downsamples]

    if not level_dimensions:
        raise ValueError("WSI has no readable pyramid levels")

    eligible_levels = [
        (level, max_dimension)
        for level, max_dimension in level_dimensions
        if max_dimension <= MAX_THUMBNAIL_MAX_DIMENSION
    ]

    if not eligible_levels:
        best_level = len(level_dimensions) - 1
        return best_level, downsamples[best_level]

    level0_max_dimension = level_dimensions[0][1]
    if level0_max_dimension <= TARGET_THUMBNAIL_MAX_DIMENSION:
        return 0, downsamples[0]

    best_level, _ = min(
        eligible_levels,
        key=lambda item: abs(item[1] - TARGET_THUMBNAIL_MAX_DIMENSION),
    )
    return best_level, downsamples[best_level]


def _read_rgb_thumbnail(
    slide: openslide.OpenSlide,
    level: int,
) -> np.ndarray:
    """Read a full low-resolution WSI level as an RGB numpy array."""
    level_dimensions = slide.level_dimensions[level]
    thumbnail_rgba = slide.read_region((0, 0), level, level_dimensions)
    thumbnail_rgb = thumbnail_rgba.convert("RGB")
    return np.asarray(thumbnail_rgb)


def _build_mask_from_rgb(thumbnail_rgb: np.ndarray) -> np.ndarray:
    """Create a cleaned binary tissue mask from an RGB thumbnail."""
    thumbnail_hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)
    saturation = thumbnail_hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    min_dimension = min(mask.shape)
    kernel_size = max(3, int(round(min_dimension * 0.01)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return (mask > 0).astype(np.uint8)


def _load_thumbnail_and_mask(wsi_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load the WSI thumbnail and compute its tissue mask."""
    resolved_wsi_path = _ensure_path_is_under(wsi_path, RAW_DIR, "wsi_path")
    if not resolved_wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {resolved_wsi_path}")

    with openslide.OpenSlide(str(resolved_wsi_path)) as slide:
        level, downsample = _select_mask_level(slide)
        thumbnail_rgb = _read_rgb_thumbnail(slide, level)

    mask = _build_mask_from_rgb(thumbnail_rgb)
    return thumbnail_rgb, mask, downsample


def generate_tissue_mask(wsi_path: Path) -> tuple[np.ndarray, float]:
    """Generate a binary tissue mask for a WSI.

    Args:
        wsi_path: Path to an .svs WSI under data/raw/.

    Returns:
        A tuple containing the binary uint8 numpy mask and the exact WSI
        pyramid downsample factor used to generate it.
    """
    _, mask, downsample = _load_thumbnail_and_mask(wsi_path)
    return mask, downsample


def visualize_mask(wsi_path: Path, output_dir: Path = DEFAULT_DEBUG_DIR) -> Path:
    """Save a side-by-side RGB thumbnail and tissue-mask debug plot."""
    resolved_output_dir = _ensure_path_is_under(
        output_dir,
        INTERIM_DIR,
        "output_dir",
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    thumbnail_rgb, mask, downsample = _load_thumbnail_and_mask(wsi_path)
    slide_name = Path(wsi_path).stem
    output_path = resolved_output_dir / f"{slide_name}_tissue_mask_debug.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(thumbnail_rgb)
    axes[0].set_title(f"RGB thumbnail ({downsample:.2f}x)")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Tissue mask")
    axes[1].axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tissue-mask debug plot for a TCGA WSI.",
    )
    parser.add_argument(
        "wsi_path",
        type=Path,
        help="Path to an .svs WSI under data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help="Debug plot output directory under data/interim/.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = visualize_mask(args.wsi_path, args.output_dir)
    print(f"Saved mask debug plot to {output_path}")


if __name__ == "__main__":
    main()
