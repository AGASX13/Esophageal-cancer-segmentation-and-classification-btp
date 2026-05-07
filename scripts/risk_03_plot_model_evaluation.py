from __future__ import annotations

import argparse
from pathlib import Path

from src.risk_engine.evaluation import plot_stage1_risk_eval
from src.risk_engine.model_xgboost import default_configs


def main() -> int:
    default_risk_cfg, _ = default_configs()

    parser = argparse.ArgumentParser(description="Generate Stage-1 risk model visualizations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_risk_cfg,
        help="Risk config YAML (default: config/risk_engine/base.yaml).",
    )
    args = parser.parse_args()

    out_dir = plot_stage1_risk_eval(args.config)
    print(f"Saved Stage-1 plots in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

