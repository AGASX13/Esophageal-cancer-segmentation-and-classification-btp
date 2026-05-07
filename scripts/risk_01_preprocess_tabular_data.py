from __future__ import annotations

import argparse
from pathlib import Path

from src.risk_engine.data_prep import default_config_path, run_preprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess risk CSV and build train/val/test splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Risk config YAML (default: config/risk_engine/base.yaml).",
    )
    args = parser.parse_args()

    out_dir = run_preprocess(args.config)
    print(f"Risk data processed into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

