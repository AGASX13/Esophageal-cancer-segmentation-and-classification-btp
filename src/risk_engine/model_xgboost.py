from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from src.common.config_loader import load_yaml
from src.common.paths import get_paths


def _load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    p = processed_dir / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing split file: {p}")
    return pd.read_csv(p)


def train_xgboost(risk_config_path: str | Path, xgb_config_path: str | Path) -> Path:
    risk_cfg = load_yaml(risk_config_path)
    xgb_cfg = load_yaml(xgb_config_path)

    seed = int(risk_cfg.get("seed", 42))
    processed_dir = Path(risk_cfg["paths"]["processed_dir"])
    model_dir = Path(risk_cfg["paths"]["model_dir"])
    artifacts_dir = Path(risk_cfg["paths"]["artifacts_dir"])

    meta_path = processed_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run preprocessing first (scripts/risk_01_preprocess_tabular_data.py)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_col = meta["target_col"]

    train_df = _load_split(processed_dir, "train")
    val_df = _load_split(processed_dir, "val")
    test_df = _load_split(processed_dir, "test")

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col].astype(int)
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col].astype(int)
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col].astype(int)

    try:
        from xgboost import XGBClassifier
    except Exception as e:  # pragma: no cover
        raise RuntimeError("xgboost is not installed in this environment") from e

    params: dict[str, Any] = dict(xgb_cfg["model"]["params"])
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", -1)

    model = XGBClassifier(**params)

    # Older/newer xgboost versions have slightly different fit signatures.
    # To stay compatible we avoid passing unsupported kwargs.
    fit_kwargs: dict[str, Any] = {}
    early = xgb_cfg.get("training", {}).get("early_stopping_rounds", None)
    eval_metric = xgb_cfg.get("training", {}).get("eval_metric", None)

    if early is not None:
        # Only pass early stopping args if supported by this version.
        # If not, we just train without early stopping.
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=int(early),
                eval_metric=eval_metric,
                verbose=False,
            )
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    def predict_proba(m, X):
        p = m.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p

    yhat = model.predict(X_test)
    prob = predict_proba(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, yhat)),
        "roc_auc": float(roc_auc_score(y_test, prob)) if len(set(y_test.tolist())) > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, yhat).tolist(),
        "classification_report": classification_report(y_test, yhat, output_dict=True),
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "target_col": target_col,
        "params": params,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_model = model_dir / "xgboost_risk.joblib"
    joblib.dump(
        {"model": model, "target_col": target_col, "feature_columns": list(X_train.columns)},
        out_model,
    )

    out_metrics = artifacts_dir / "xgboost_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return out_model


def default_configs() -> tuple[Path, Path]:
    p = get_paths()
    return (
        p.config / "risk_engine" / "base.yaml",
        p.config / "risk_engine" / "xgboost_default.yaml",
    )

