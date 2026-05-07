## EC-CAD (Esophageal Cancer Computer Aided Diagnosis)

This repository is structured as a **3-stage pipeline**:

- **Stage-1 (Risk Engine)**: tabular risk prediction (XGBoost) to decide whether a pathology WSI is needed.
- **Stage-2 (Segmentation)**: train a segmentation model (PanNuke) and generate masks on WSIs.
- **Stage-3 (Classification)**: slide-level classification (CLAM / MIL) using WSI patches/features.

You can run tasks either:
- **via CMake orchestration targets** (recommended for repeatability), or
- **direct Python commands**.

---

## Project structure (important folders)

- **`data/`**: all datasets (raw/interim/processed)
- **`config/`**: YAML configs for each stage
- **`models/`**: saved model artifacts (checkpoints / joblib)
- **`experiments/`**: metrics, logs, run outputs
- **`scripts/`**: CLI entrypoints you run
- **`src/`**: Python source code

---

## Setup

### Create + activate virtual environment

```bat
cd /d C:\Users\sj428\Desktop\main-project-btp
python -m venv .venv
.venv\Scripts\activate
```

### Install Python dependencies

```bat
pip install -r requirements.txt
```

Notes:
- Some packages can be OS/CPU-specific (especially OpenSlide + some scientific wheels).
- If any install fails, install the failing package separately (we can fix pins later).

---

## Where to put datasets

### Stage-1 Risk dataset (Kaggle CSV)

Copy your CSV to:

- `data/raw/risk/data.csv`

If your CSV uses different column names, update:
- `config/risk_engine/base.yaml` → `columns:`

### TCGA WSI dataset (organized by class folder)

Put slides into:

- `data/raw/tcga_esca_wsi/cancer/`  (label = 1)
- `data/raw/tcga_esca_wsi/normal/`  (label = 0)

Supported extensions in the label generator:
`.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`

### PanNuke (segmentation) dataset

You said your PanNuke folder looks like:

- `data/raw/pannuke/fold1/images + masks`
- `data/raw/pannuke/fold2/images + masks`
- `data/raw/pannuke/fold3/images + masks`

That matches the code. Default training config uses **fold2**.

---

## Commands (CMake orchestration)

### Initialize a CMake build folder once

```bat
cmake -S . -B build
```

### Generate slide labels CSV from folder structure

```bat
cmake --build build --target gen_slide_labels
```

This writes:
- `data/raw/tcga_esca_annotations/slide_labels.csv`

### Run Stage-1 risk pipeline (preprocess + train)

```bat
cmake --build build --target risk_all
```

### Train Stage-2 segmentation (PanNuke U-Net)

```bat
cmake --build build --target seg_train
```

Outputs:
- Processed splits: `data/processed/risk_engine/` (`train.csv`, `val.csv`, `test.csv`, `meta.json`)
- Model: `models/risk_engine/xgboost_risk.joblib`
- Metrics: `experiments/risk_engine/xgboost_metrics.json`

### List available CMake targets

```bat
cmake --build build --target help
```

---

## Commands (direct Python)

All commands below assume you activated `.venv`.

### Generate slide labels

```bat
.venv\Scripts\python scripts\tcga_02_generate_slide_labels.py
```

### Stage-1: preprocess risk CSV into train/val/test splits

```bat
.venv\Scripts\python scripts\risk_01_preprocess_tabular_data.py --config config\risk_engine\base.yaml
```

### Stage-1: train XGBoost risk model

```bat
.venv\Scripts\python scripts\risk_02_train_xgboost_model.py ^
  --risk-config config\risk_engine\base.yaml ^
  --xgb-config config\risk_engine\xgboost_default.yaml
```

### Stage-2: train U-Net on PanNuke

```bat
.venv\Scripts\python scripts\segmentation_legacy_train_unet_pannuke.py --config config\segmentation\pannuke_unet_resnet34_base.yaml
```

---

## What to do next

- When your **PanNuke** download finishes, place it into `data/raw/pannuke/` (we’ll add training scripts next).
- When your **WSIs** finish downloading, place them under `data/raw/tcga_esca_wsi/cancer/` and `.../normal/`,
  then run `gen_slide_labels`.
- After that, we’ll add **segmentation + patching + CLAM** scripts and register them as new CMake targets.

