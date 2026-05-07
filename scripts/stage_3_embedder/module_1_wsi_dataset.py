"""Module 1: WSI patch dataset for Stage 3 embedding extraction."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WsiPatchDataset(Dataset):
    """Read Level 0 WSI patches using HDF5 coordinate vault entries."""

    def __init__(self, wsi_path: str | Path, h5_path: str | Path) -> None:
        self.wsi_path = Path(wsi_path)
        self.h5_path = Path(h5_path)

        if not self.wsi_path.is_file():
            raise FileNotFoundError(f"WSI not found: {self.wsi_path}")
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"H5 vault not found: {self.h5_path}")

        self.slide = None

        # Open, extract into memory, and immediately close the h5 file.
        with h5py.File(self.h5_path, "r") as f:
            if "coords" not in f:
                raise KeyError(f"Dataset 'coords' not found in {self.h5_path}")
            
            coords_ds = f["coords"]
            self.coords = coords_ds[:]  # Copies data to a NumPy array
            self.patch_size = int(coords_ds.attrs["patch_size"])

        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"Expected coords shape (N, 2), got {self.coords.shape}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.slide is None:
            self.slide = openslide.OpenSlide(str(self.wsi_path))

        x, y = self.coords[idx]
        patch = self.slide.read_region(
            (int(x), int(y)),
            0,
            (self.patch_size, self.patch_size),
        )
        patch_rgb = patch.convert("RGB")
        return self.transform(patch_rgb)

    def close(self) -> None:
        if hasattr(self, "slide") and self.slide is not None:
            self.slide.close()
            self.slide = None

    def __del__(self) -> None:
        self.close()
