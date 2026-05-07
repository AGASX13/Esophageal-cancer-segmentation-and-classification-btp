"""Module 1: MIL dataset and dataloaders for slide-level classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENSOR_DIR = PROJECT_ROOT / "data" / "processed" / "feature_tensors"
DEFAULT_WSI_ROOT_DIR = PROJECT_ROOT / "data" / "raw" / "tcga_esca_wsi"


def _base_name(path: Path) -> str:
    stem = path.name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return stem


class MilTensorDataset(Dataset):
    """MIL slide dataset over precomputed [N, 768] feature tensors."""

    def __init__(self, tensor_dir: Path, wsi_root_dir: Path) -> None:
        self.tensor_dir = Path(tensor_dir)
        self.wsi_root_dir = Path(wsi_root_dir)

        if not self.tensor_dir.is_dir():
            raise FileNotFoundError(f"Tensor directory not found: {self.tensor_dir}")
        if not self.wsi_root_dir.is_dir():
            raise FileNotFoundError(f"WSI root directory not found: {self.wsi_root_dir}")

        self.samples: list[tuple[Path, int]] = []
        cancerous_count = 0
        non_cancerous_count = 0

        for pt_path in sorted(self.tensor_dir.glob("*.pt")):
            base_name = _base_name(pt_path)
            matches = list(self.wsi_root_dir.rglob(f"{base_name}*.svs"))

            if not matches:
                continue
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple WSI matches found for {pt_path.name}: {matches}"
                )

            resolved_svs = str(matches[0]).lower()
            if "non_cancerous" in resolved_svs:
                label = 0
                non_cancerous_count += 1
            elif "cancerous" in resolved_svs:
                label = 1
                cancerous_count += 1
            else:
                continue

            self.samples.append((pt_path, label))

        print(
            f"Found {cancerous_count} cancerous and "
            f"{non_cancerous_count} non-cancerous slides."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pt_path, label = self.samples[idx]
        features = torch.load(pt_path, weights_only=True)
        features = features.to(dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return features, label_tensor


def get_mil_dataloaders(
    tensor_dir: Path,
    wsi_root_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build reproducible train/val dataloaders with batch_size=1."""
    dataset = MilTensorDataset(tensor_dir=tensor_dir, wsi_root_dir=wsi_root_dir)
    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError("No valid MIL samples found.")

    val_size = int(round(dataset_len * val_split))
    val_size = min(max(val_size, 1), dataset_len - 1) if dataset_len > 1 else 0
    train_size = dataset_len - val_size

    generator = torch.Generator().manual_seed(seed)
    if val_size == 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_mil_dataloaders(
        tensor_dir=DEFAULT_TENSOR_DIR,
        wsi_root_dir=DEFAULT_WSI_ROOT_DIR,
    )
    first_features, first_labels = next(iter(train_loader))
    print(f"Train batch features shape: {tuple(first_features.shape)}")
    print(f"Train batch labels shape: {tuple(first_labels.shape)}")
    first_val_features, first_val_labels = next(iter(val_loader))
    print(f"Val batch features shape: {tuple(first_val_features.shape)}")
    print(f"Val batch labels shape: {tuple(first_val_labels.shape)}")
