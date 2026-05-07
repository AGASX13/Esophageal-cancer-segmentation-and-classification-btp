"""Stage 2 YOLO filter: keep only coordinates with neoplastic detections.

This script reads a Stage 1 coordinate vault (.h5), extracts patch images from a
WSI in batches, runs YOLOv8 segmentation inference, and writes a filtered .h5
vault containing only coordinates that satisfy the configured class-confidence
threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "yolo_filtered_vaults"
DEFAULT_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "experiment_2"
    / "yolov8m_pannuke_esophagus_exp2_augmented_best.pt"
)


def _load_stage1_vault(vault_path: Path) -> tuple[np.ndarray, int, float | None]:
    """Load coordinates and metadata from a Stage 1 coordinate vault."""
    with h5py.File(vault_path, "r") as h5_file:
        if "coords" not in h5_file:
            raise KeyError(f"Dataset 'coords' not found in {vault_path}")

        coords_ds = h5_file["coords"]
        coords = coords_ds[:]
        patch_size = int(coords_ds.attrs["patch_size"])
        downsample_factor = (
            float(coords_ds.attrs["downsample_factor"])
            if "downsample_factor" in coords_ds.attrs
            else None
        )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords shape (N, 2), got {coords.shape}")

    return coords, patch_size, downsample_factor


def _contains_target_class(
    result,
    target_class_idx: int,
    conf_threshold: float,
) -> bool:
    """Return True if target class appears with confidence >= threshold."""
    if result.boxes is None or len(result.boxes) == 0:
        return False

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()

    target_confidences = confidences[classes == target_class_idx]
    if target_confidences.size == 0:
        return False

    return float(np.max(target_confidences)) >= conf_threshold


class YoloWsiDataset(Dataset):
    """Lazy OpenSlide dataset for threaded patch extraction."""

    def __init__(self, wsi_path: Path, coords: np.ndarray, patch_size: int) -> None:
        self.wsi_path = Path(wsi_path)
        self.coords = coords
        self.patch_size = int(patch_size)
        # Keep this lazy to avoid Windows worker pickling issues.
        self.slide = None

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        if self.slide is None:
            self.slide = openslide.OpenSlide(str(self.wsi_path))

        x, y = self.coords[idx]
        patch = self.slide.read_region(
            (int(x), int(y)),
            0,
            (self.patch_size, self.patch_size),
        ).convert("RGB")
        patch_array = np.array(patch)
        return patch_array, idx

    def __del__(self) -> None:
        if self.slide is not None:
            self.slide.close()
            self.slide = None


def _yolo_collate_fn(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[list[np.ndarray], list[int]]:
    """Split dataset tuples into YOLO image list and index list."""
    batch_images = [item[0] for item in batch]
    batch_indices = [item[1] for item in batch]
    return batch_images, batch_indices


def filter_coordinate_vault_with_yolo(
    wsi_path: Path,
    stage1_vault_path: Path,
    yolo_weights_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    conf_threshold: float = 0.5,
    target_class_idx: int = 0,
) -> Path:
    """Run batched YOLO inference and save filtered coordinates to a new .h5."""
    if not wsi_path.is_file():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")
    if not stage1_vault_path.is_file():
        raise FileNotFoundError(f"Stage 1 vault not found: {stage1_vault_path}")
    if not yolo_weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")

    coords, patch_size, downsample_factor = _load_stage1_vault(stage1_vault_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(yolo_weights_path))

    surviving_coords: list[tuple[int, int]] = []
    dataset = YoloWsiDataset(wsi_path=wsi_path, coords=coords, patch_size=patch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_yolo_collate_fn,
    )

    for batch_images, batch_indices in tqdm(
        loader,
        desc="Stage 2 YOLO filtering",
        unit="batch",
    ):
        results = model(
            batch_images,
            conf=0.5,
            iou=0.45,
            max_det=50,
            half=True,
            device=0,
            verbose=False,
        )

        for coord_idx, result in zip(batch_indices, results):
            if _contains_target_class(result, target_class_idx, conf_threshold):
                coord = coords[coord_idx]
                surviving_coords.append((int(coord[0]), int(coord[1])))

    surviving_array = np.asarray(surviving_coords, dtype=np.int32)
    if surviving_array.size == 0:
        surviving_array = surviving_array.reshape(0, 2)

    output_path = output_dir / f"{stage1_vault_path.stem}.h5"
    with h5py.File(output_path, "w") as h5_file:
        coords_ds = h5_file.create_dataset(
            "coords",
            data=surviving_array,
            dtype=np.int32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        coords_ds.attrs["patch_size"] = int(patch_size)
        if downsample_factor is not None:
            coords_ds.attrs["downsample_factor"] = float(downsample_factor)
        coords_ds.attrs["yolo_conf_threshold"] = float(conf_threshold)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 YOLO filtering on a single WSI + Stage 1 .h5 vault and "
            "save surviving coordinates to data/interim/yolo_filtered_vaults/."
        )
    )
    parser.add_argument("wsi_path", type=Path, help="Path to input WSI (.svs).")
    parser.add_argument(
        "stage1_vault_path",
        type=Path,
        help="Path to Stage 1 coordinate vault (.h5).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to YOLOv8 .pt weights (default: {DEFAULT_WEIGHTS_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of patches to infer per batch.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to keep target class detections.",
    )
    parser.add_argument(
        "--target-class-idx",
        type=int,
        default=0,
        help="Class index to keep (default assumes 0=Neoplastic).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for threaded patch extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_path = filter_coordinate_vault_with_yolo(
        wsi_path=args.wsi_path,
        stage1_vault_path=args.stage1_vault_path,
        yolo_weights_path=args.weights,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_threshold=args.conf_threshold,
        target_class_idx=args.target_class_idx,
    )
    print(f"Saved YOLO-filtered coordinate vault to {output_path}")


if __name__ == "__main__":
    main()
