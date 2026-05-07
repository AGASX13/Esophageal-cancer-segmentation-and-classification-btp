from __future__ import annotations

import argparse
from pathlib import Path

from src.segmentation.train_pannuke import default_config_path, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train U-Net segmentation on PanNuke folds.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Segmentation config YAML (default: config/segmentation/pannuke_unet_resnet34_base.yaml).",
    )
    args = parser.parse_args()

    ckpt = train(args.config)
    print(f"Saved best checkpoint to: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

