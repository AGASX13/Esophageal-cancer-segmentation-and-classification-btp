from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.risk_engine.inference import load_risk_model, predict_risk_from_row


router = APIRouter(tags=["risk"])


class RiskRequest(BaseModel):
    # This expects already-preprocessed feature keys matching training features.
    # We'll improve this to raw clinical fields + preprocessing later.
    features: dict[str, Any] = Field(default_factory=dict)


class RiskResponse(BaseModel):
    label: int
    probability: float


@router.post("/risk/predict", response_model=RiskResponse)
def predict(req: RiskRequest) -> RiskResponse:
    try:
        bundle = load_risk_model()
        pred = predict_risk_from_row(req.features, bundle)
        return RiskResponse(label=pred.label, probability=pred.probability)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {e}") from e

