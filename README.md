# Overview
A production-style deep learning pipeline for multi-class classification of jute crop pests using pretrained CNN backbones (EfficientNet, VGG16, DenseNet). This project demonstrates:
- Transfer learning with frozen ImageNet backbones
- tf.data input pipelines with augmentation & caching
- Structured experiment logging
- Automated evaluation with precision / recall / F1 / ROC-AUC
- Model comparison across multiple architectures
- Reproducible training with fixed random seeds
- Unit tests using pytest


# Project Structure
The repository is organized as follows.  

```
jute-pest-image-classification/
│
├── cache/                 # Disk cache for training dataset (not tracked in git - created at runtime)
├── data/                  # Dataset root (not tracked in git - create and extract data after download separately)
│   └── Jute_Pest_Dataset/
│       ├── train/
│       ├── val/
│       └── test/
│
├── logs/                  # Per-run and per-backbone logs (not tracked in git - created at runtime)
├── models/                # Saved best models (not tracked in git - created at runtime)
├── notebook/              # Narrative style notebook for the project
├── results/               # Metrics, plots, reports, comparisons (not tracked in git - created at runtime)
│
├── src/
│   ├── config.py          # Global configuration
│   ├── dataset.py         # Dataset creation & preprocessing
│   ├── main.py            # Training + evaluation entry point
│   ├── models.py          # Model architectures
│   ├── utils.py           # Logging, metrics, plotting helpers
│   └── tests.py           # Pytest test suite
│
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```


# Data sources
This project uses the Jute Pest Dataset for training, validation, and testing which can be downloaded as a zip file from https://archive.ics.uci.edu/dataset/920/jute+pest+dataset

Extract the contents to the data directory in project root.
The data directory in project root should look like this after extraction.
```
data/
└── Jute_Pest_Dataset/
    ├── train/
    │   ├── Beet Armyworm/
    │   ├── Black Hairy/
    │   ├── Cutworm/
    │   ├── ... (other 17 classes)
    │
    ├── val/
    │   ├── Beet Armyworm/
    │   ├── Black Hairy/
    │   ├── Cutworm/
    │   ├── ... (same class folders as train)
    │
    └── test/
        ├── Beet Armyworm/
        ├── Black Hairy/
        ├── Cutworm/
        ├── ... (same class folders as train)
```
Each class folder should contain the corresponding pest images.

**Note:** 
- The code loads images using tf.keras.utils.image_dataset_from_directory(), so correct folder structure is essential.
- Labels are inferred from folder names.
- Your validation and test sets must include all class subfolders present in training.


# Models
Implemented backbones:
- EfficientNetB0
- VGG16
- DenseNet201

Each model:
- Loads pretrained ImageNet weights
- Freezes backbone layers
- Adds a custom classification head:
	- Dense + L2 regularization
	- Batch normalization
	- ReLU
	- Dropout
	- Softmax output
Optimizer: Adam
Loss: Categorical Cross-Entropy


# Data Pipeline
Implemented in dataset.py:
- Resizing + normalization
- Training-only data augmentation:
	- Random crop
	- Flip
	- Rotation
	- Translation
	- Zoom
	- Contrast
- tf.data pipeline with:
	- Disk caching for training only
	- No caching for validation / test
	- Prefetching with AUTOTUNE

This avoids Windows file-locking issues while preserving training speed.


# Configuration
Key parameters (from config.py):
- Image size: 224×224
- Batch size: 16
- Epochs: 30 (max)
- Early stopping patience: 5
- Learning rate: 1e-4
- Dropout: 0.20
- L2 regularization: 1e-4
- Random seed: 42
- Backbones to train:
	- EfficientNetB0
	- VGG16
	- DenseNet201


# Installation
Create environment using:
```
conda create -n jute python=3.10
conda activate jute
```
Then:
- Install runtime dependencies via: pip install -r requirements.txt
- Install dev dependencies via: pip install -r requirements-dev.txt


# Jupyter Notebook
The complete narrative analysis is available in:
`notebook/analysis.ipynb`
This notebook presents the project as a story with explanations, plots, and interpretations.


# Training
From project root run:

`python src/main.py`

The script will:
1. Load datasets
2. Train each backbone
3. Save best model weights
4. Save training curves
5. Generate classification reports
6. Compute macro precision / recall / F1 / ROC-AUC
7. Export comparison tables (CSV + JSON)


# Outputs
For each backbone:
```
results/<backbone_name>/
├── accuracy.png
├── loss.png
├── <backbone>_Train_classification_report.txt
├── <backbone>_Val_classification_report.txt
├── <backbone>_Test_classification_report.txt
└── history.json
```
Global:
```
results/
├── models_comparison.csv
└── models_comparison.json
```
Models:
```
models/best_<backbone>.keras
```
Logs:
```
logs/
├── EfficientNetB0.log
├── VGG16.log
└── DenseNet201.log
```


# Logging Design
- One run logger for console output
- One child logger per backbone writing to its own log file
- Timestamped structured format


# Running Tests 
From root run:

`pytest src/tests.py`

This runs lightweight functional tests. Includes:
- Configuration sanity checks
- Model forward pass tests
- Preprocessing function validation
- Optional dataset shape tests
- Logging helper tests
- Dataset tests are skipped automatically if dataset is not present.