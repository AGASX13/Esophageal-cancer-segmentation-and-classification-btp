from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from src.common.config_loader import load_yaml


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_stage1_risk_eval(config_path: str | Path) -> Path:
    """
    Create basic visualizations for Stage-1 risk model:
    - ROC curve
    - confusion matrix heatmap
    - feature importance bar plot
    """
    cfg = load_yaml(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])

    reports_dir = _ensure_dir(Path("reports") / "risk_engine")

    meta = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))
    target_col = meta["target_col"]

    test_df = pd.read_csv(processed_dir / "test.csv")
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    bundle = joblib.load(model_dir / "xgboost_risk.joblib")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Align columns just in case
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    prob = model.predict_proba(X_test)[:, 1]
    y_pred = (prob >= 0.5).astype(int)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax, name="XGBoost")
    ax.set_title("Stage-1 Risk Model ROC")
    roc_path = reports_dir / "risk_roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage-1 Risk Confusion Matrix")
    cm_path = reports_dir / "risk_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Feature importance
    importance = np.array(getattr(model, "feature_importances_", np.zeros(len(feature_columns))))
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=importance[order],
        y=np.array(feature_columns)[order],
        orient="h",
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Stage-1 Risk Feature Importances")
    ax.set_xlabel("Importance")
    fi_path = reports_dir / "risk_feature_importance.png"
    fig.tight_layout()
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)

    # Save a small JSON manifest of generated plots
    manifest = {
        "roc_curve": str(roc_path),
        "confusion_matrix": str(cm_path),
        "feature_importance": str(fi_path),
    }
    (reports_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return reports_dir

