from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.common.config_loader import load_yaml
from src.common.paths import get_paths


@dataclass(frozen=True)
class PreparedSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target_col: str


def _seed_everything(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _find_col(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    # case-insensitive fallback
    lowered = {c.lower(): c for c in df.columns}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise KeyError(f"Column '{name}' not found. Available columns: {list(df.columns)[:30]}...")


def add_gerd_feature(
    df: pd.DataFrame,
    *,
    bmi_col: str,
    smoking_col: str,
    out_col: str,
    seed: int,
    base: float,
    bmi_weight: float,
    smoking_bonus: float,
) -> pd.DataFrame:
    """
    Add a synthetic GERD feature using a simple probability model.
    Output is binary {0,1}.
    """
    rng = _seed_everything(seed)

    bmi = pd.to_numeric(df[bmi_col], errors="coerce").fillna(df[bmi_col].median()).to_numpy()
    smoking_raw = df[smoking_col]

    # Robust smoking indicator:
    # - numeric: >0 -> smoker
    # - text: contains 'yes', 'current', 'smok' -> smoker; contains 'no', 'never' -> non-smoker
    if pd.api.types.is_numeric_dtype(smoking_raw):
        is_smoker = (pd.to_numeric(smoking_raw, errors="coerce").fillna(0) > 0).to_numpy()
    else:
        s = smoking_raw.astype(str).str.lower()
        is_smoker = (
            s.str.contains("yes|current|smok", regex=True) & ~s.str.contains("no|never", regex=True)
        ).to_numpy()

    prob = base + bmi_weight * (bmi - 25.0) + smoking_bonus * is_smoker.astype(float)
    prob = _clamp(prob, 0.0, 0.95)
    df[out_col] = (rng.random(len(df)) < prob).astype(int)
    return df


def add_risk_target(
    df: pd.DataFrame,
    *,
    target_col: str,
    gerd_col: str,
    smoking_col: str,
    bmi_col: str,
    obesity_bmi_threshold: float,
    gerd_weight: int,
    smoking_weight: int,
    obesity_weight: int,
    high_risk_threshold: int,
) -> pd.DataFrame:
    gerd = pd.to_numeric(df[gerd_col], errors="coerce").fillna(0).astype(int)
    bmi = pd.to_numeric(df[bmi_col], errors="coerce").fillna(df[bmi_col].median())

    smoking_raw = df[smoking_col]
    if pd.api.types.is_numeric_dtype(smoking_raw):
        smoker = (pd.to_numeric(smoking_raw, errors="coerce").fillna(0) > 0).astype(int)
    else:
        s = smoking_raw.astype(str).str.lower()
        smoker = (
            s.str.contains("yes|current|smok", regex=True) & ~s.str.contains("no|never", regex=True)
        ).astype(int)

    obese = (bmi >= obesity_bmi_threshold).astype(int)
    score = gerd_weight * gerd + smoking_weight * smoker + obesity_weight * obese

    # Binary target: 1 = high risk, 0 = low risk
    df[target_col] = (score > high_risk_threshold).astype(int)
    df["_risk_score"] = score.astype(int)
    return df


def preprocess_and_split(config: dict[str, Any]) -> PreparedSplits:
    seed = int(config.get("seed", 42))
    cols = config["columns"]
    fe = config["feature_engineering"]
    split_cfg = config["split"]
    pp = config["preprocess"]

    raw_csv = Path(config["paths"]["raw_csv"])
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw risk CSV not found at '{raw_csv}'. Put your Kaggle CSV there or edit config."
        )

    df = pd.read_csv(raw_csv)

    age_col = _find_col(df, cols["age"])
    bmi_col = _find_col(df, cols["bmi"])
    gender_col = _find_col(df, cols["gender"])
    smoking_col = _find_col(df, cols["smoking"])
    alcohol_col = _find_col(df, cols["alcohol"])

    # Keep a focused set of columns first (you can expand later)
    keep = [age_col, bmi_col, gender_col, smoking_col, alcohol_col]
    df = df[keep].copy()

    if pp.get("drop_na", True):
        df = df.dropna()

    gerd_col = fe.get("gerd_column", "GERD")
    if fe.get("create_gerd", True) and gerd_col not in df.columns:
        inj = fe["gerd_injection"]
        df = add_gerd_feature(
            df,
            bmi_col=bmi_col,
            smoking_col=smoking_col,
            out_col=gerd_col,
            seed=seed,
            base=float(inj["base"]),
            bmi_weight=float(inj["bmi_weight"]),
            smoking_bonus=float(inj["smoking_bonus"]),
        )

    rs = fe["risk_scoring"]
    target_col = rs.get("target_column", "Esophageal_Risk")
    df = add_risk_target(
        df,
        target_col=target_col,
        gerd_col=gerd_col,
        smoking_col=smoking_col,
        bmi_col=bmi_col,
        obesity_bmi_threshold=float(fe["obesity_bmi_threshold"]),
        gerd_weight=int(rs["gerd_weight"]),
        smoking_weight=int(rs["smoking_weight"]),
        obesity_weight=int(rs["obesity_weight"]),
        high_risk_threshold=int(rs["high_risk_threshold"]),
    )

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    stratify = y if bool(split_cfg.get("stratify", True)) else None
    train_ratio = float(split_cfg["train"])
    val_ratio = float(split_cfg["val"])
    test_ratio = float(split_cfg["test"])
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1.0 - train_ratio), random_state=seed, stratify=stratify
    )

    # Split tmp into val/test
    tmp_size = val_ratio + test_ratio
    val_frac_of_tmp = val_ratio / tmp_size

    stratify_tmp = y_tmp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=(1.0 - val_frac_of_tmp),
        random_state=seed,
        stratify=stratify_tmp,
    )

    train = X_train.copy()
    train[target_col] = y_train.values
    val = X_val.copy()
    val[target_col] = y_val.values
    test = X_test.copy()
    test[target_col] = y_test.values

    # One-hot encode categoricals
    if pp.get("one_hot_encode", True):
        cat_cols = [c for c in train.columns if train[c].dtype == "object"]
        train = pd.get_dummies(train, columns=cat_cols, drop_first=False)
        val = pd.get_dummies(val, columns=cat_cols, drop_first=False)
        test = pd.get_dummies(test, columns=cat_cols, drop_first=False)

        # Align columns across splits
        train_cols = train.columns
        val = val.reindex(columns=train_cols, fill_value=0)
        test = test.reindex(columns=train_cols, fill_value=0)

    # Scale numeric features (excluding target)
    if pp.get("scale_numeric", True):
        feature_cols = [c for c in train.columns if c != target_col]
        scaler = MinMaxScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols])
        val[feature_cols] = scaler.transform(val[feature_cols])
        test[feature_cols] = scaler.transform(test[feature_cols])

        processed_dir = Path(config["paths"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "scaler.json").write_text(
            json.dumps({"type": "MinMaxScaler"}, indent=2), encoding="utf-8"
        )

    return PreparedSplits(train=train, val=val, test=test, target_col=target_col)


def run_preprocess(config_path: str | Path) -> Path:
    cfg = load_yaml(config_path)
    splits = preprocess_and_split(cfg)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    splits.train.to_csv(train_path, index=False)
    splits.val.to_csv(val_path, index=False)
    splits.test.to_csv(test_path, index=False)

    meta = {
        "target_col": splits.target_col,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "processed_dir": str(processed_dir),
    }
    (processed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return processed_dir


def default_config_path() -> Path:
    return get_paths().config / "risk_engine" / "base.yaml"

