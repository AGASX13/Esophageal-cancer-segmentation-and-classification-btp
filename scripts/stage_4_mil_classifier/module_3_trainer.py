"""Module 3: MIL trainer with validation and early stopping."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from module_1_mil_dataset import get_mil_dataloaders
from module_2_attention_net import AttentionMIL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENSOR_DIR = PROJECT_ROOT / "data" / "processed" / "feature_tensors"
DEFAULT_WSI_ROOT = PROJECT_ROOT / "data" / "raw" / "tcga_esca_wsi"
DEFAULT_MODEL_SAVE_DIR = PROJECT_ROOT / "models"


def _binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    correct = (preds == labels).float().sum().item()
    return float(correct / labels.numel())


def train_mil_classifier(
    tensor_dir: Path = DEFAULT_TENSOR_DIR,
    wsi_root: Path = DEFAULT_WSI_ROOT,
    num_epochs: int = 50,
    learning_rate: float = 2e-4,
    patience: int = 5,
) -> None:
    """Train AttentionMIL classifier with early stopping on validation loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_mil_dataloaders(
        tensor_dir=tensor_dir,
        wsi_root_dir=wsi_root,
    )

    model = AttentionMIL().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    model_save_dir = DEFAULT_MODEL_SAVE_DIR
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_dir / "best_mil_classifier.pth"

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for features, labels in tqdm(train_loader, desc=f"Train {epoch}/{num_epochs}", unit="slide"):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_acc_sum += _binary_accuracy_from_logits(logits, labels)
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_acc = train_acc_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Val {epoch}/{num_epochs}", unit="slide"):
                features = features.to(device)
                labels = labels.to(device)

                logits, _ = model(features)
                loss = criterion(logits, labels)

                val_loss_sum += float(loss.item())
                val_acc_sum += _binary_accuracy_from_logits(logits, labels)
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        val_acc = val_acc_sum / max(val_batches, 1)

        print(
            f"Epoch [{epoch}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved improved model to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break


if __name__ == "__main__":
    train_mil_classifier()
