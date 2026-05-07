# EC-CAD (Esophageal Cancer Computer Aided Diagnosis) - Complete Project Summary

## Project Overview

**EC-CAD** is a 3-stage computer vision pipeline for automated esophageal cancer diagnosis from whole slide images (WSI). It combines **risk prediction**, **tissue segmentation**, and **slide-level classification** to provide a complete diagnostic workflow.

### Architecture
```
Stage 1: Risk Engine (XGBoost)
   ↓ (tabular risk prediction from patient data)
   ↓ (filters which WSIs need analysis)
   ↓
Stage 2: Segmentation (YOLOv8-seg)
   ↓ (instance segmentation of tissue patches)
   ↓ (generates tissue masks on WSIs)
   ↓
Stage 3: Classification (CLAM/MIL)
   ↓ (slide-level classification using patch features)
   ↓
Final Diagnosis & Risk Assessment
```

---

## 📁 Directory Structure & Contents

### Root Level Files
- **`CMakeLists.txt`**: CMake build configuration for orchestrating all pipeline stages and tasks
- **`requirements.txt`**: Python package dependencies (numpy, pandas, torch, scikit-learn, etc.)
- **`README.md`**: Main project documentation
- **`data.yaml`**: YOLO dataset configuration (train/val paths)
- **`models/segmentation/`**: Pre-trained and experiment-specific segmentation weights

---

## 📂 Main Directories

### 1. **`data/`** - Dataset Storage & Organization
Contains all raw, interim, and processed datasets.

#### Subdirectories:
- **`data/raw/`**: Original/unprocessed data
  - `pannuke/`: PanNuke segmentation dataset (fold1, fold2, fold3 with images + masks)
  - `tcga_esca_wsi/`: Whole slide images organized by class (cancer/, normal/)
  - `tcga_esca_annotations/`: TCGA annotation files
  - `risk/`: CSV files for risk prediction (tabular patient data)
  - `wsi_custom/`: Custom WSI dataset

- **`data/interim/`**: Partially processed datasets
  - `pannuke_patches/`: Preprocessed PanNuke patches
  - `risc_cleaned/`: Cleaned RISC dataset
  - `tcga_tissue_masks/`: Generated tissue masks from WSIs

- **`data/processed/`**: Final processed data ready for model training
  - `risk_engine/`: Processed risk model data with meta.json
  - `segmentation/`: Processed segmentation datasets (YOLO format)
  - `tcga_features/`: Extracted features from WSIs
  - `tcga_patches/`: Extracted patches from WSIs
  - `wsi_level_outputs/`: Aggregated WSI-level predictions

---

### 2. **`src/`** - Python Source Code
Core library modules organized by pipeline stage.

#### **`src/common/`** - Shared Utilities
- `config_loader.py`: YAML configuration loading & parsing
- `paths.py`: Path management and dataset directory utilities
- `__init__.py`: Package initialization

#### **`src/risk_engine/`** - Stage 1: Risk Prediction
XGBoost-based tabular risk prediction.

**Files:**
- `data_prep.py`: Data preprocessing, feature engineering, train/val/test splits
- `model_xgboost.py`: XGBoost model training, hyperparameter tuning, model saving
- `inference.py`: Risk prediction on new patient data
- `evaluation.py`: Model evaluation metrics (AUC, accuracy, calibration curves)

**Purpose:** Predicts patient risk from tabular features (age, comorbidities, etc.) to filter which WSIs require detailed analysis.

#### **`src/segmentation/`** - Stage 2: Tissue Segmentation
UNet and YOLOv8-based instance segmentation.

**Files:**
- `pannuke_dataset.py`: Dataset loader for PanNuke (5-cell class segmentation)
- `train_pannuke.py`: UNet training on PanNuke data
- `unet_models.py`: UNet architecture definitions and utilities
- `__init__.py`: Package initialization

**Purpose:** Segments tissue types (neoplastic, inflammatory, connective, dead cells) in WSI patches.

#### **`src/classification/`** - Stage 3: Slide Classification
Multiple Instance Learning (MIL) based classification.

**Files:**
- `clam_model.py`: CLAM (Clustering-constrained Attention Multiple Instance Learning) implementation
- `__init__.py`: Package initialization

**Purpose:** Aggregates patch-level features into slide-level cancer classification using attention mechanisms.

#### **`src/pipeline/`** - Orchestration
- `run_risk_then_wsi.py`: End-to-end pipeline coordinator that runs all 3 stages sequentially

---

### 3. **`config/`** - Configuration Files (YAML)
Centralized configuration for each pipeline stage.

#### Structure:
- **`config/risk_engine/`**:
  - `base.yaml`: Risk model base configuration (data columns, train/val split ratio)
  - `xgboost_default.yaml`: XGBoost hyperparameters (learning rate, max depth, n_estimators)

- **`config/classification/`**: Classification stage configs

- **`config/pipeline/`**: Full pipeline orchestration configs

- **`config/segmentation/`**: Segmentation stage configs

**Usage:** Models load YAML configs at runtime for reproducibility.

---

### 4. **`scripts/`** - CLI Entry Points & Utilities
Executable Python scripts for running individual pipeline components. ⭐ **DETAILED BREAKDOWN BELOW**

---

### 5. **`models/`** - Trained Model Artifacts
Saved model weights and checkpoints.

- **`models/segmentation/`**: Trained segmentation models (YOLOv8, UNet checkpoints)
- **`models/classification/`**: Classification model weights
- **`models/risk_engine/`**: 
  - `xgboost_risk.joblib`: Serialized XGBoost model

---

### 6. **`experiments/`** - Experimental Results & Logs
Outputs from model training runs.

- **`experiments/risk_engine/`**: 
  - `xgboost_metrics.json`: Risk model evaluation metrics

- **`experiments/segmentation/`**: Segmentation training logs

- **`experiments/classification/`**: Classification training logs

- **`experiments/ablation_studies/`**: Results from ablation studies

---

### 7. **`runs/`** - YOLOv8 Training Output
YOLOv8-specific training artifacts.

- **`runs/segment/`**: YOLOv8 segmentation training runs
  - `esophagus_train/`: Main training run
  - `esophagus_train2/`: Secondary training run
  - `esophagus_val/`: Validation outputs
  - `esophagus_val_probe/`, `esophagus_val_probe2/`: Probe/debugging runs

Each contains weights, metrics, and visualizations.

---

### 8. **`data/processed/segmentation/yolo_exp1_base/` & `data/processed/segmentation/yolo_exp1_augmented/`** - Training Datasets (YOLO Format)

#### **`data/processed/segmentation/yolo_exp1_base/`** - Original dataset
- `images/train/`: Training patch images
- `images/val/`: Validation patch images
- `labels/train/`: YOLO-format segmentation labels (normalized polygons)
- `labels/val/`: Validation labels

#### **`data/processed/segmentation/yolo_exp1_augmented/`** - Augmented training data
- `images/train/`: Augmented patches (rotations, flips, brightness changes)
- `labels/train/`: Corresponding augmented labels

**Purpose:** Training data for YOLOv8-seg instance segmentation.

---

### 9. **`artifacts/`** - Evaluation & Reporting
Final outputs and summaries.

- **`artifacts/sanity_checks/`**: Validation results
- **`artifacts/evaluations/`**:
  - `final_report/evaluation_summary.txt`: Final evaluation report
  - `training_curves/`: Loss & accuracy curves across epochs

---

### 10. **`notebooks/`** - Jupyter Notebooks (EDA & Prototyping)
Exploratory and experimental notebooks organized by stage.

- **`notebooks/risk_engine/`**: Risk prediction exploration
- **`notebooks/segmentation/`**: Segmentation model experiments
- **`notebooks/classification/`**: Classification model development
- **`notebooks/misc/`**: Miscellaneous experiments

---

### 11. **`frontend/`** - Web UI (React)
React-based web interface for inference and visualization.

- **`frontend/src/`**:
  - `api/`: API client calls to backend
  - `components/`: React components (UI elements)
  - `hooks/`: Custom React hooks
  - `pages/`: Page components
  - `styles/`: CSS styling

- **`frontend/public/`**: Static assets (HTML, images)

---

### 12. **`backend/`** - REST API Server
Flask/FastAPI-based backend for serving models.

- **`backend/app/`**:
  - `main.py`: FastAPI/Flask app initialization
  - `core/`: Business logic
  - `ml/`: ML model loading & inference wrappers
  - `models/`: Pydantic data models
  - `routes/`: API endpoints (e.g., `risk.py` for risk prediction)
  - `services/`: Helper services (DB, file upload, etc.)

- **`backend/tests/`**: Unit tests for API
- **`backend/README.md`**: Backend setup instructions

---

### 13. **`reports/`** - Generated Reports & Visualizations
Output reports from pipeline runs.

- **`reports/cdss_demo/`**: Clinical decision support demo outputs
- **`reports/risk_engine/`**:
  - `plots_manifest.json`: Manifest of generated plots
- **`reports/segmentation/`**: Segmentation visualizations
- **`reports/classification/`**: Classification results

---

### 14. **`docs/`** - Documentation
- **`docs/paper-for-segmentation/`**: Research paper or technical documentation for segmentation

---

## 🔧 Scripts Directory - Detailed Breakdown

### Critical Scripts Overview

The `scripts/` folder contains 11 standalone Python executables for running individual pipeline tasks. All scripts follow the pattern: `def main() -> int` for CLI usage.

---

#### **1. Stage-1: Risk Engine Scripts**

##### **`risk_02_train_xgboost_model.py`**
- **Purpose**: Train XGBoost model for patient risk prediction
- **Input**: Processed risk data (train/val splits from `data/processed/risk_engine/`)
- **Output**: Trained model saved to `models/risk_engine/xgboost_risk.joblib`
- **Config Files Used**: `config/risk_engine/base.yaml`, `config/risk_engine/xgboost_default.yaml`
- **Usage**: `python scripts/risk_02_train_xgboost_model.py --risk-config <path> --xgb-config <path>`
- **Dependencies**: src.risk_engine.model_xgboost

##### **`risk_01_preprocess_tabular_data.py`**
- **Purpose**: Preprocess raw risk CSV (tabular patient data) into train/val/test splits
- **Input**: `data/raw/risk/data.csv`
- **Output**: Processed splits in `data/processed/risk_engine/`
- **Config**: `config/risk_engine/base.yaml` (column names, split ratios)
- **Usage**: `python scripts/risk_01_preprocess_tabular_data.py`
- **Dependencies**: src.risk_engine.data_prep

##### **`risk_03_plot_model_evaluation.py`**
- **Purpose**: Generate evaluation plots (ROC curves, calibration, confusion matrix) for risk model
- **Input**: Trained XGBoost model + test data
- **Output**: PNG/PDF plots to `reports/risk_engine/`
- **Usage**: `python scripts/risk_03_plot_model_evaluation.py`
- **Dependencies**: src.risk_engine.evaluation, matplotlib, seaborn

---

#### **2. Stage-2: Segmentation Scripts (PanNuke)**

##### **`module_01_prepare_yolo_dataset.py`**
- **Purpose**: Convert raw PanNuke dataset (.npy files) → YOLO-format segmentation labels
- **Input**: 
  - PanNuke raw folds: `data/raw/pannuke/fold1,2,3/` (images.npy + masks.npy)
  - Filters to **Esophagus patches only**
- **Output**: 
  - `data/processed/segmentation/yolo_exp1_base/images/train/`, `data/processed/segmentation/yolo_exp1_base/images/val/` (YOLO format)
  - `data/processed/segmentation/yolo_exp1_base/labels/train/`, `data/processed/segmentation/yolo_exp1_base/labels/val/` (normalized polygon labels)
- **Mapping**: PanNuke cell class channels 0-4 → YOLO class IDs (Neoplastic, Inflammatory, Connective, Dead, Other)
- **Train/Val Split**: Uses sklearn.train_test_split (default 80/20)
- **Usage**: `python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_01_prepare_yolo_dataset.py --fold 2`
- **Key Logic**: 
  - Filters to esophagus patches
  - Converts instance masks → normalized polygon coordinates
  - Generates `classes.txt` with class names
- **Dependencies**: cv2, numpy, sklearn, tqdm

##### **`module_03_augment_training_split.py`**
- **Purpose**: Apply data augmentation (rotation, flip, brightness, elastic distortion) to training set
- **Input**: `data/processed/segmentation/yolo_exp1_base/images/train/` + `data/processed/segmentation/yolo_exp1_base/labels/train/`
- **Output**: Augmented patches → `data/processed/segmentation/yolo_exp1_augmented/images/train/` + `data/processed/segmentation/yolo_exp1_augmented/labels/train/`
- **Augmentation Techniques**: 
  - Random rotation (±15°)
  - Horizontal/vertical flips
  - Brightness/contrast adjustments
  - Elastic distortion
- **Usage**: `python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_03_augment_training_split.py --num-aug 3`
- **Dependencies**: albumentations, cv2, numpy

##### **`segmentation_legacy_train_unet_pannuke.py`**
- **Purpose**: Train UNet model on PanNuke dataset (alternative to YOLOv8)
- **Input**: PanNuke dataset in `data/processed/segmentation/yolo_exp1_base/` or `data/processed/segmentation/`
- **Output**: Model checkpoint to `models/segmentation/`
- **Config**: `config/segmentation/` (learning rate, epochs, batch size)
- **Usage**: `python scripts/segmentation_legacy_train_unet_pannuke.py`
- **Dependencies**: src.segmentation.train_pannuke, torch, torchvision

---

#### **3. Stage-2: Segmentation Scripts (YOLOv8)**

##### **`module_04_train_yolov8s_segmentation.py`**
- **Purpose**: Train YOLOv8-seg instance segmentation model on esophagus patches
- **Input**: 
  - Training data: `data/processed/segmentation/yolo_exp1_augmented/images/train/` + `data/processed/segmentation/yolo_exp1_augmented/labels/train/`
  - Validation data: `data/processed/segmentation/yolo_exp1_base/images/val/` + `data/processed/segmentation/yolo_exp1_base/labels/val/`
  - Model: Pre-trained `models/segmentation/pretrained/yolov8s_seg_coco_pretrained.pt`
- **Output**: 
  - Trained model weights: `models/segmentation/experiment_1/`
  - Metrics: stored in runs directory
- **Key Process**:
  1. Writes `data.yaml` at project root
  2. Initializes YOLOv8 model from Ultralytics
  3. Trains for N epochs with validation on each epoch
  4. Saves best model weights
- **Config**: Image size (640x640), epochs (50-100), batch size (16-32)
- **Usage**: `python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_04_train_yolov8s_segmentation.py --epochs 50 --batch 16`
- **Dependencies**: ultralytics, torch, torchvision

##### **`module_05_evaluate_yolov8s_segmentation.py`**
- **Purpose**: Evaluate trained YOLOv8-seg model on test set, compute metrics
- **Input**: 
  - Trained model from `models/segmentation/experiment_1/`
  - Test data: `data/processed/segmentation/yolo_exp1_base/images/val/` + `data/processed/segmentation/yolo_exp1_base/labels/val/`
- **Output**: 
  - Metrics: mAP50, mAP50-95, instance precision/recall
  - Results saved to `experiments/segmentation/` or `artifacts/evaluations/`
- **Metrics Computed**:
  - Mean Average Precision (mAP) at IOU thresholds 0.5 and 0.5-0.95
  - Class-wise precision and recall
  - Loss curves during training
- **Usage**: `python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_05_evaluate_yolov8s_segmentation.py --weights <path>`
- **Dependencies**: ultralytics, torch, numpy

##### **`module_02_validate_label_overlays.py`**
- **Purpose**: Visual validation - overlay predicted masks on original images
- **Input**: 
  - Trained YOLOv8-seg model
  - Test images from `data/processed/segmentation/yolo_exp1_base/images/val/`
- **Output**: 
  - PNG images with overlaid segmentation masks
  - Saved to `artifacts/evaluations/` or `reports/segmentation/`
- **Purpose**: Qualitative inspection of segmentation quality
- **Usage**: `python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_02_validate_label_overlays.py --out-dir <path>`
- **Dependencies**: ultralytics, cv2, numpy, matplotlib

---

#### **4. Visualization & Reporting Scripts**

##### **`module_06_plot_training_curves.py`**
- **Purpose**: Plot training curves (loss, mAP, precision, recall) from YOLOv8 training
- **Input**: Training runs directory (e.g., `runs/segment/esophagus_train/`)
- **Output**: 
  - PNG plots: training/validation loss, mAP over epochs
  - Saved to `artifacts/evaluations/training_curves/`
- **Visualizations**: 
  - Box loss (segmentation mask loss)
  - Cls loss (classification loss)
  - mAP50 and mAP50-95
- **Usage**: `python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_06_plot_training_curves.py --results-csv <path>`
- **Dependencies**: matplotlib, pandas, seaborn

##### **`tcga_02_generate_slide_labels.py`**
- **Purpose**: Generate slide-level labels CSV from WSI folder structure
- **Input**: WSI folders organized by class (cancer/, normal/) at `data/raw/tcga_esca_wsi/`
- **Output**: 
  - `data/raw/tcga_esca_annotations/slide_labels.csv`
  - Columns: slide_name, label (0=normal, 1=cancer), filepath
- **Supported Extensions**: `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`
- **Usage**: `python scripts/tcga_02_generate_slide_labels.py --input-dir <wsi-folder>`
- **Dependencies**: Path, pandas, tqdm

---

## 🔄 Execution Flow

### CMake-Based Orchestration (Recommended)
```bash
# Initialize CMake build system
cmake -S . -B build

# Generate slide labels
cmake --build build --target gen_slide_labels

# Run Stage-1 (Risk)
cmake --build build --target run_risk_pipeline

# Run Stage-2 (Segmentation)
cmake --build build --target train_segmentation

# Full pipeline
cmake --build build --target run_full_pipeline
```

### Direct Python Execution
```bash
# Activate venv
.venv\Scripts\activate

# Stage 1: Risk
python scripts/risk_01_preprocess_tabular_data.py
python scripts/risk_02_train_xgboost_model.py
python scripts/risk_03_plot_model_evaluation.py

# Stage 2: Segmentation Experiment 1
python scripts/segmentation_experiments/experiment_01_baseline_yolov8s/module_99_orchestrator.py

# Stage 2: Segmentation Experiment 2
python scripts/segmentation_experiments/experiment_02_augmented_yolov8m/module_99_orchestrator.py
```

---

## 📊 Data Flow Diagram

```
Raw Data:
├─ data/raw/risk/data.csv (patient tabular data)
├─ data/raw/tcga_esca_wsi/ (WSI slides by class)
├─ data/raw/pannuke/ (PanNuke folds with masks)
└─ data/raw/tcga_esca_annotations/ (metadata)

                ↓

Preprocessing Scripts:
├─ risk_01_preprocess_tabular_data.py → data/processed/risk_engine/
├─ tcga_02_generate_slide_labels.py → data/raw/tcga_esca_annotations/slide_labels.csv
└─ experiment_01_baseline_yolov8s/module_01_prepare_yolo_dataset.py → data/processed/segmentation/yolo_exp1_base/ (YOLO format)

                ↓

Data Augmentation:
└─ experiment_01_baseline_yolov8s/module_03_augment_training_split.py → data/processed/segmentation/yolo_exp1_augmented/

                ↓

Training Scripts:
├─ risk_02_train_xgboost_model.py → models/risk_engine/xgboost_risk.joblib
├─ experiment_01_baseline_yolov8s/module_04_train_yolov8s_segmentation.py → models/segmentation/ + runs/segment/
└─ segmentation_legacy_train_unet_pannuke.py → models/segmentation/

                ↓

Evaluation Scripts:
├─ experiment_01_baseline_yolov8s/module_05_evaluate_yolov8s_segmentation.py → experiments/
├─ experiment_01_baseline_yolov8s/module_02_validate_label_overlays.py → artifacts/evaluations/
├─ risk_03_plot_model_evaluation.py → reports/risk_engine/
└─ experiment_01_baseline_yolov8s/module_06_plot_training_curves.py → artifacts/evaluations/training_curves/

                ↓

Final Outputs:
├─ reports/ (plots, tables, summaries)
├─ artifacts/ (evaluations, visualizations)
├─ experiments/ (metrics, logs)
└─ models/ (inference-ready weights)
```

---

## 🛠️ Technology Stack

### ML/Deep Learning
- **PyTorch**: Deep learning framework
- **Ultralytics YOLOv8**: Instance segmentation
- **scikit-learn**: XGBoost, utilities
- **pandas, numpy**: Data processing

### Computer Vision
- **OpenCV (cv2)**: Image processing
- **Albumentations**: Image augmentation
- **Pillow**: Image I/O

### Web Framework
- **React**: Frontend UI
- **Flask/FastAPI**: REST API backend
- **Pydantic**: Data validation

### DevOps
- **CMake**: Build orchestration
- **Docker**: Containerization (optional)
- **YAML**: Configuration management

### Visualization
- **Matplotlib, Seaborn**: Plots and curves
- **Plotly**: Interactive visualizations

---

## 📝 Configuration Files Summary

| File | Purpose |
|------|---------|
| `config/risk_engine/base.yaml` | Risk model data columns, train/val split ratio |
| `config/risk_engine/xgboost_default.yaml` | XGBoost hyperparameters (LR, max_depth, n_estimators) |
| `config/classification/*.yaml` | Classification stage configs |
| `config/pipeline/*.yaml` | Full pipeline orchestration configs |
| `config/segmentation/*.yaml` | Segmentation stage hyperparameters |
| `data.yaml` | YOLO dataset format specification |

---

## 🚀 Quick Reference for Another AI

**To understand the project:**
1. Start with `README.md` for architecture overview
2. Review `scripts/` folder for entry points and execution order
3. Check `src/` module structure for implementation details
4. Examine `config/` YAML files for hyperparameters
5. Review `data/` directory structure for dataset organization

**Key Concepts:**
- **3-Stage Pipeline**: Risk → Segmentation → Classification
- **CMake-based Orchestration**: Reproducible, automated runs
- **Multiple Backends**: YOLOv8 (main), UNet (alternative), XGBoost (risk)
- **YAML Configs**: All hyperparameters externalized
- **Jupyter Notebooks**: For experimentation and EDA
- **REST API**: Backend serves models for inference
- **React Frontend**: Web UI for clinicians

---

## 📦 Key Artifacts & Checkpoints

| Type | Location | Purpose |
|------|----------|---------|
| Models | `models/` | Trained weights ready for inference |
| Runs | `runs/segment/` | YOLOv8 training history & artifacts |
| Reports | `reports/` | Human-readable plots & summaries |
| Artifacts | `artifacts/` | Evaluation results & visualizations |
| Experiments | `experiments/` | Metrics JSON, logs |
| Data | `data/processed/` | Ready-to-train preprocessed data |

