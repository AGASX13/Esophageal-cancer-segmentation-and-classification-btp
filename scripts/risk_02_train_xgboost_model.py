from __future__ import annotations

import argparse
from pathlib import Path

from src.risk_engine.model_xgboost import default_configs, train_xgboost


def main() -> int:
    default_risk_cfg, default_xgb_cfg = default_configs()

    parser = argparse.ArgumentParser(description="Train XGBoost risk model from processed splits.")
    parser.add_argument(
        "--risk-config",
        type=Path,
        default=default_risk_cfg,
        help="Risk config YAML (default: config/risk_engine/base.yaml).",
    )
    parser.add_argument(
        "--xgb-config",
        type=Path,
        default=default_xgb_cfg,
        help="XGBoost config YAML (default: config/risk_engine/xgboost_default.yaml).",
    )
    args = parser.parse_args()

    model_path = train_xgboost(args.risk_config, args.xgb_config)
    print(f"Saved XGBoost model to: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

