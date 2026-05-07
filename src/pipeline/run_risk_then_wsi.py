from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.risk_engine.inference import load_risk_model, predict_risk_from_row


def run_full_pipeline(patient_features: dict[str, Any], wsi_path: str | Path | None = None) -> dict[str, Any]:
    """
    End-to-end orchestration entrypoint.

    Right now:
    - runs Stage-1 risk prediction from already-preprocessed features
    Later:
    - if high risk and a WSI is provided: run segmentation + CLAM and return combined report.
    """
    bundle = load_risk_model()
    pred = predict_risk_from_row(patient_features, bundle)

    result: dict[str, Any] = {
        "risk": asdict(pred),
        "wsi_provided": wsi_path is not None,
        "wsi_analysis": None,
    }

    # TODO: integrate segmentation + CLAM inference
    return result

