# Scripts Index

Root-level scripts use a domain-prefixed action name:

```text
<domain>_<step>_<action>_<object>.py
```

## Risk Engine

- `risk_01_preprocess_tabular_data.py`: prepare train/val/test data for the risk model.
- `risk_02_train_xgboost_model.py`: train and save the XGBoost risk model.
- `risk_03_plot_model_evaluation.py`: generate risk-model evaluation plots.

## TCGA WSI Utilities

- `tcga_01_download_esca_wsi_slides.py`: download TCGA-ESCA WSI slides.
- `tcga_02_generate_slide_labels.py`: create slide-level labels from TCGA WSI folders.

## Segmentation Utilities

- `segmentation_legacy_train_unet_pannuke.py`: legacy/alternate U-Net PanNuke training entrypoint.
- `stage_1_wsi_tiling/`: ordered WSI tiling modules.
- `segmentation_experiments/`: ordered YOLO segmentation experiment modules.
