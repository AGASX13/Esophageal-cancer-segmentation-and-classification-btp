from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common.paths import get_paths


@dataclass(frozen=True)
class RiskPrediction:
    label: int
    probability: float


def load_risk_model(model_path: str | Path | None = None) -> dict[str, Any]:
    """
    Loads the trained XGBoost risk model bundle saved by `risk_02_train_xgboost_model.py`.
    """
    if model_path is None:
        model_path = get_paths().models / "risk_engine" / "xgboost_risk.joblib"
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Risk model not found: {p}")
    return joblib.load(p)


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    if p.ndim == 2 and p.shape[1] > 1:
        return p[:, 1]
    return p.reshape(-1)


def predict_risk_from_row(row: dict[str, Any], model_bundle: dict[str, Any]) -> RiskPrediction:
    """
    Predict risk from a single already-preprocessed feature row.

    Notes:
    - This expects keys that match the trained feature columns.
    - For a full production path, your API should reuse the same preprocessing
      pipeline as training (one-hot + scaling). We'll wire that up later.
    """
    model = model_bundle["model"]
    feature_cols: list[str] = model_bundle["feature_columns"]

    X = pd.DataFrame([row])
    X = X.reindex(columns=feature_cols, fill_value=0)

    prob = float(_predict_proba(model, X)[0])
    label = int(prob >= 0.5)
    return RiskPrediction(label=label, probability=prob)

