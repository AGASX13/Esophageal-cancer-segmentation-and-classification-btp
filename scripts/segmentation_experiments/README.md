# Segmentation Experiments

Segmentation experiments are organized as ordered modules so each experiment can
be rerun, audited, or resumed without guessing script order.

```text
scripts/segmentation_experiments/
├── experiment_01_baseline_yolov8s/
└── experiment_02_augmented_yolov8m/
```

Each experiment folder uses:

```text
module_01_...
module_02_...
module_03_...
module_99_orchestrator.py
```

## Experiment 1: Baseline YOLOv8s

Data locations:

- Base YOLO dataset: `data/processed/segmentation/yolo_exp1_base/`
- Augmented YOLO dataset: `data/processed/segmentation/yolo_exp1_augmented/`

```powershell
python scripts\segmentation_experiments\experiment_01_baseline_yolov8s\module_99_orchestrator.py
```

Modules:

1. `module_01_prepare_yolo_dataset.py`
2. `module_02_validate_label_overlays.py`
3. `module_03_augment_training_split.py`
4. `module_04_train_yolov8s_segmentation.py`
5. `module_05_evaluate_yolov8s_segmentation.py`
6. `module_06_plot_training_curves.py`

## Experiment 2: Augmented YOLOv8m

Data locations:

- Refined YOLO dataset: `data/processed/segmentation/yolo_exp2_refined/`
- Augmented YOLO dataset: `data/processed/segmentation/yolo_exp2_augmented/`
- Hard-negative source: `data/raw/hard_negatives/nct_crc_stroma/`

```powershell
python scripts\segmentation_experiments\experiment_02_augmented_yolov8m\module_99_orchestrator.py
```

Modules:

1. `module_01_smooth_segmentation_polygons.py`
2. `module_02_inject_hard_negatives.py`
3. `module_03_apply_medical_augmentations.py`
4. `module_04_train_yolov8m_segmentation.py`
5. `module_05_evaluate_yolov8m_segmentation.py`
6. `module_06_predict_and_filter_masks.py`

Both orchestrators support `--start-at`, `--stop-after`, and `--dry-run`.
