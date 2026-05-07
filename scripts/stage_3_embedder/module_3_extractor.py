"""Module 3: single-slide embedding extractor using Phikon."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from module_1_wsi_dataset import WsiPatchDataset
from module_2_encoder import build_pathology_encoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "feature_tensors"


def _base_name(path: Path) -> str:
    stem = path.name
    while Path(stem).suffix:
        stem = Path(stem).stem
    return stem


def extract_slide_embeddings(
    wsi_path: Path,
    h5_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Path:
    """Extract CLS embeddings for one slide and save as .pt tensor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = WsiPatchDataset(wsi_path=wsi_path, h5_path=h5_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_pathology_encoder(device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_base_name(wsi_path)}.pt"

    embedding_chunks: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Embedding {_base_name(wsi_path)}", unit="batch"):
                batch = batch.to(device, non_blocking=True)
                if device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        outputs = model(batch)
                else:
                    outputs = model(batch)

                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embedding_chunks.append(cls_embeddings.cpu())
    finally:
        dataset.close()

    if embedding_chunks:
        embeddings = torch.cat(embedding_chunks, dim=0)
    else:
        embeddings = torch.empty((0, 768), dtype=torch.float32)

    torch.save(embeddings, output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Stage 3 Phikon embeddings for a single WSI and H5 vault.",
    )
    parser.add_argument("wsi_path", type=Path, help="Path to input WSI (.svs).")
    parser.add_argument("h5_path", type=Path, help="Path to YOLO-filtered .h5 vault.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output .pt tensor (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = extract_slide_embeddings(
        wsi_path=args.wsi_path,
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved embedding tensor to {output_path}")


if __name__ == "__main__":
    main()
