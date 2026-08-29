# Jute Pest Image Classification

A transfer-learning image classifier that identifies 17 species of jute crop pests from photographs, served via a production FastAPI + Docker inference API.

## Overview

Three ImageNet-pretrained backbones (EfficientNetB0, VGG16, DenseNet201) were trained and evaluated using a two-stage transfer-learning strategy — frozen-backbone training followed by partial fine-tuning — with the best-performing model selected automatically and packaged for deployment.

| |                                                   |
|---|---------------------------------------------------|
| **Task** | 17-class multi-class image classification         |
| **Selected model** | EfficientNetB0 (fine-tuned)                       |
| **Test macro AUC** | 0.99999                                           |
| **Test accuracy** | 99%                                               |
| **Serving** | FastAPI + Docker, single-model inference endpoint |

## Dataset

17 jute pest species, split into train/validation/test sets:

Beet Armyworm · Black Hairy · Cutworm · Field Cricket · Jute Aphid · Jute Hairy · Jute Red Mite · Jute Semilooper · Jute Stem Girdler · Jute Stem Weevil · Leaf Beetle · Mealybug · Pod Borer · Scopula Emissaria · Termite · Termite odontotermes (Rambur) · Yellow Mite

## Approach

Each backbone was trained in two stages:

1. **Initial training** — backbone frozen (ImageNet weights), only a custom classification head trained (`Dense(256) → BatchNorm → ReLU → Dropout → Dense(17)`), with data augmentation (flip, rotation, translation, zoom, contrast) applied ahead of the backbone.
2. **Fine-tuning** — the final 20% of backbone layers unfrozen (BatchNorm layers kept frozen) and retrained at a lower learning rate.

Early stopping and `ReduceLROnPlateau` were used at both stages, monitoring validation loss. The stage (initial vs. fine-tuned) that achieved the lower validation loss was retained for each backbone — fine-tuning won for all three:

| Backbone | Initial val_loss | Fine-tuned val_loss | Stage retained |
|---|---|---|---|
| EfficientNetB0 | 0.3640 | **0.3216** | Fine-tuned |
| VGG16 | 0.4692 | **0.3966** | Fine-tuned |
| DenseNet201 | 0.3760 | **0.3352** | Fine-tuned |

## Model Comparison

Macro-averaged metrics across all three backbones (best stage per backbone):

| Model | Val Precision | Val Recall | Val F1 | Val AUC | Test Precision | Test Recall | Test F1 | Test AUC |
|---|---|---|---|---|---|---|---|---|
| **EfficientNetB0** | 0.927 | 0.933 | 0.928 | **0.9958** | 0.989 | 0.989 | 0.989 | **0.99999** |
| VGG16 | 0.879 | 0.881 | 0.876 | 0.9782 | 0.986 | 0.987 | 0.986 | 0.99997 |
| DenseNet201 | 0.888 | 0.873 | 0.876 | 0.9952 | 0.982 | 0.981 | 0.981 | 0.99975 |

**EfficientNetB0 (fine-tuned)** was selected as the deployment model based on validation macro AUC.

### Per-class performance (selected model, test set)

99% overall accuracy, with all 17 classes achieving F1 ≥ 0.92. The validation set is where the model's actual generalization limits show: classes with the fewest validation examples (Pod Borer: 3, Jute Semilooper: 5, Jute Hairy: 8) had the lowest F1 scores (0.67, 0.73, 0.82 respectively), consistent with limited support rather than a systematic weakness in the model.

## Project Structure

```
jute-pest-image-classification/
├── artifacts/          # Deployment package: selected model + manifest.json (git-tracked)
├── data/                # Train/val/test image directories (gitignored)
├── models/              # All trained checkpoints, full manifest (gitignored)
├── results/             # Per-backbone plots, classification reports, comparison CSV/JSON (git-tracked)
├── logs/                # Training logs (gitignored)
├── src/
│   ├── app.py           # FastAPI application — serving endpoints
│   ├── config.py        # Paths, hyperparameters, backbone list
│   ├── dataset.py        # tf.data pipeline construction
│   ├── inference.py      # Model loading, preprocessing, prediction logic
│   ├── main.py           # Training orchestration (all backbones)
│   ├── models.py         # Model architecture, backbone/preprocessing lookup
│   └── utils.py          # Logging, evaluation, manifest generation
├── tests/                # pytest suite (31 tests: API, inference, training pipeline)
├── Dockerfile
├── requirements.txt          # Full project (training + serving)
├── requirements-serve.txt    # Serving-only subset (used by Dockerfile)
└── requirements-dev.txt      # Test tooling
```

The `artifacts/` directory holds only the single selected model checkpoint and its manifest — generated automatically at the end of training — keeping the deployment footprint independent of how many candidate models were trained.

## API

Three endpoints, backed by a model loaded once at application startup:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — reports whether the model loaded successfully |
| `POST` | `/predict` | Upload an image (JPEG/PNG, ≤5MB) → returns predicted class, confidence, and top-3 probabilities |
| `GET` | `/model/info` | Returns the deployed model's backbone, training stage, and validation/test AUC |

Interactive API docs are available at `/docs` once the service is running.

## Running Locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
python src/main.py          # retrains all backbones and regenerates artifacts/
uvicorn app:app --app-dir src --reload
```

## Running with Docker

```bash
docker build -t jute-pest-api .
docker run -p 8000:8000 jute-pest-api
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Testing

```bash
pytest tests/ -v
```

31 tests covering the FastAPI endpoints, inference/preprocessing logic, and the training pipeline's model-building, evaluation, and manifest-generation utilities.

## Engineering Notes

- **Manifest-driven deployment**: model selection happens once, at training time, based on validation macro AUC. The serving layer reads the selected backbone from `artifacts/manifest.json` rather than hardcoding it — retraining and redeploying a new winner requires no code changes.
- **Train/serve preprocessing consistency**: backbone-specific normalization (e.g., EfficientNet's `preprocess_input`) is applied inside the model graph itself, not duplicated in the serving code — eliminating a common source of silent accuracy degradation in deployed CV models.
- **Two-stage transfer learning**: partial fine-tuning (final 20% of backbone layers, BatchNorm excluded) improved validation loss for every backbone tested, and the initial-vs-fine-tuned comparison is made automatically rather than assumed.