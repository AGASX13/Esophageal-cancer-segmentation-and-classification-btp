"""Module 4: MIL evaluator with ROC and confusion matrix plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from tqdm import tqdm

from module_1_mil_dataset import DEFAULT_TENSOR_DIR, DEFAULT_WSI_ROOT_DIR, get_mil_dataloaders
from module_2_attention_net import AttentionMIL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_mil_classifier.pth"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def evaluate_model() -> None:
    """Evaluate trained MIL model and save publication-ready plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _, val_loader = get_mil_dataloaders(
        tensor_dir=DEFAULT_TENSOR_DIR,
        wsi_root_dir=DEFAULT_WSI_ROOT_DIR,
        seed=42,
    )

    model = AttentionMIL().to(device)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    all_labels: list[float] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Evaluating", unit="slide"):
            features = features.to(device)
            labels = labels.to(device)

            logits, _ = model(features)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.cpu().numpy().flatten().astype(int).tolist())

    labels_np = np.asarray(all_labels, dtype=np.float32)
    probs_np = np.asarray(all_probs, dtype=np.float32)
    preds_np = np.asarray(all_preds, dtype=np.int32)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(labels_np, probs_np)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", linewidth=1.5, linestyle="--", label="Chance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MIL Classifier ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(labels_np.astype(int), preds_np.astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Cancerous", "Cancerous"],
        yticklabels=["Non-Cancerous", "Cancerous"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)

    # Optional sklearn display object for compatibility with requested imports.
    _ = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Cancerous", "Cancerous"],
    )

    print(f"AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Saved ROC curve to {roc_path}")
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    evaluate_model()
