from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.common.config_loader import load_yaml
from src.common.paths import get_paths
from src.segmentation.pannuke_dataset import build_dataloaders
from src.segmentation.unet_models import build_unet_resnet34


def train(config_path: str | Path) -> Path:
    """
    Training entrypoint for PanNuke U-Net segmentation.

    Minimal implementation:
    - Loads images/masks from one fold (fold1/fold2/fold3)
    - Random train/val split
    - Trains U-Net (ResNet34 encoder) using CrossEntropyLoss
    - Saves best checkpoint by lowest val loss
    """
    cfg = load_yaml(config_path)
    paths = get_paths()

    # Deps check early for clearer errors
    try:
        import torch
        import torch.nn as nn
        from torch.optim import Adam
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for segmentation training. Install `torch` first.") from e

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    pannuke_root = Path(cfg["data"]["pannuke_root"])
    fold = str(cfg["data"]["fold"])
    image_size = cfg["data"].get("image_size", None)
    val_split = float(cfg["data"].get("val_split", 0.2))

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])

    num_classes = int(cfg["model"]["num_classes"])

    train_loader, val_loader = build_dataloaders(
        pannuke_root=pannuke_root,
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=int(image_size) if image_size is not None else None,
        val_split=val_split,
        seed=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_resnet34(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)

    out_model_dir = paths.root / cfg["output"]["model_dir"]
    out_artifacts_dir = paths.root / cfg["output"]["artifacts_dir"]
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_path = out_model_dir / "unet_resnet34_best.pth"

    history = {"train_loss": [], "val_loss": [], "epochs": epochs, "fold": fold}

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            bsz = x.shape[0]
            train_loss += float(loss.item()) * bsz
            n_train += bsz

        train_loss = train_loss / max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                bsz = x.shape[0]
                val_loss += float(loss.item()) * bsz
                n_val += bsz

        val_loss = val_loss / max(n_val, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "image_size": image_size,
                    "fold": fold,
                    "config": cfg,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                },
                best_path,
            )

        print(f"[ep {ep:03d}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} best={best_val:.4f}")

    (out_artifacts_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path


def default_config_path() -> Path:
    return get_paths().config / "segmentation" / "pannuke_unet_resnet34_base.yaml"

