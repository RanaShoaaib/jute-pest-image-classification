from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_ROOT = ROOT_DIR / "data" / "Jute_Pest_Dataset"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
TEST_DIR  = DATA_ROOT / "test"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)       # works for all chosen backbones
INPUT_SHAPE = IMAGE_SIZE + (3,)

BATCH_SIZE = 16
EPOCHS = 25
FINETUNE_EPOCHS = 10
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
LR_REDUCTION_FACTOR = 0.5
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.20
PATIENCE = 5
LR_PATIENCE = 3
UNFREEZE_LAYERS_FRAC = 0.2

# which models to train
BACKBONE_NAMES = [
    "EfficientNetB0",
    "VGG16",
    "DenseNet201",
]

# for reproducibility
SEED = 42