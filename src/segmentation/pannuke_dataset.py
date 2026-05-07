from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PanNukeLayout:
    """
    Expected minimal folder layout (you can adapt later):

      data/raw/pannuke/
        fold1/
          images/
          masks/
        fold2/
          images/
          masks/
        fold3/
          images/
          masks/
    """

    root: Path
    fold: str

    @property
    def images_dir(self) -> Path:
        return self.root / self.fold / "images"

    @property
    def masks_dir(self) -> Path:
        return self.root / self.fold / "masks"


def assert_pannuke_layout(root: Path, fold: str) -> PanNukeLayout:
    layout = PanNukeLayout(root=root, fold=fold)
    if not layout.images_dir.exists():
        raise FileNotFoundError(f"Missing PanNuke images dir: {layout.images_dir}")
    if not layout.masks_dir.exists():
        raise FileNotFoundError(f"Missing PanNuke masks dir: {layout.masks_dir}")
    return layout


def torch_required() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for segmentation stage. Install `torch` first.") from e


def _list_image_files(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


class PanNukeSegmentationDataset:
    """
    Minimal torch Dataset for PanNuke fold images + masks.

    Assumptions:
    - `images/<stem>.*` has a matching `masks/<stem>.*`
    - mask is single-channel with integer class ids (preferred)
    """

    def __init__(self, pannuke_root: Path, fold: str, image_size: int | None = None):
        torch_required()
        from torch.utils.data import Dataset  # noqa: F401

        self.layout = assert_pannuke_layout(pannuke_root, fold)
        self.image_paths = _list_image_files(self.layout.images_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.layout.images_dir}")

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import torch

        img_path = self.image_paths[idx]
        mask_path = self.layout.masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            # try any extension in masks dir
            candidates = list(self.layout.masks_dir.glob(img_path.stem + ".*"))
            if not candidates:
                raise FileNotFoundError(f"Mask not found for {img_path.name} in {self.layout.masks_dir}")
            mask_path = candidates[0]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # If mask is RGB, try to collapse to 1 channel.
        if mask.ndim == 3:
            # common case: all channels identical
            if np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 1], mask[..., 2]):
                mask = mask[..., 0]
            else:
                # fallback: use first channel (better than crashing; you can refine later)
                mask = mask[..., 0]

        if self.image_size is not None:
            s = int(self.image_size)
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        mask_t = torch.from_numpy(mask.astype(np.int64))  # (H,W) class ids
        return img_t, mask_t


def build_dataloaders(
    pannuke_root: Path,
    fold: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    image_size: int | None = None,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Simple random split into train/val from a single fold.
    """
    torch_required()
    import torch
    from torch.utils.data import DataLoader, random_split

    ds = PanNukeSegmentationDataset(pannuke_root=pannuke_root, fold=fold, image_size=image_size)
    n = len(ds)
    n_val = int(round(n * float(val_split)))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    return train_loader, val_loader

