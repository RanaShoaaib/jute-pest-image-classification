import os
import pytest
import tensorflow as tf
import config
from models import build_transfer_model, get_preprocess_fn

try:
    from dataset import create_datasets
    DATASET_AVAILABLE = True
except Exception:
    DATASET_AVAILABLE = False


# -----------------------------
# Config / directory tests
# -----------------------------

def test_required_directories_exist():
    assert config.LOGS_DIR.exists()
    assert config.CACHE_DIR.exists()
    assert config.RESULTS_DIR.exists()
    assert config.MODELS_DIR.exists()


def test_seed_defined():
    assert hasattr(config, "SEED")
    assert isinstance(config.SEED, int)


# -----------------------------
# Model tests
# -----------------------------

@pytest.mark.parametrize("backbone", config.BACKBONE_NAMES)
def test_model_build_and_forward(backbone):
    num_classes = 3
    model = build_transfer_model(backbone_name=backbone, input_shape=config.INPUT_SHAPE, num_classes=num_classes)

    assert isinstance(model, tf.keras.Model)

    # dummy forward pass
    x = tf.random.uniform((2, *config.INPUT_SHAPE))
    y = model(x)

    assert y.shape == (2, num_classes)


@pytest.mark.parametrize("backbone", config.BACKBONE_NAMES)
def test_preprocess_function_exists(backbone):
    fn = get_preprocess_fn(backbone)
    assert callable(fn)


# -----------------------------
# Dataset tests (optional)
# -----------------------------

@pytest.mark.skipif(
    not DATASET_AVAILABLE or not hasattr(config, "DATA_ROOT") or not config.DATA_ROOT.exists(),
    reason="Dataset not available on this machine",
)

def test_dataset_shapes_and_classes():
    train_ds, val_ds, test_ds, num_classes, class_names = create_datasets()

    # one batch
    x_batch, y_batch = next(iter(train_ds))

    assert x_batch.ndim == 4  # (B, H, W, C)
    assert x_batch.shape[0] > 0
    assert x_batch.shape[-1] == 3

    # labels can be int (B,) or one-hot (B, C)
    assert y_batch.ndim in (1, 2)
    if y_batch.ndim == 1: # int labels
        assert len(class_names) == num_classes
    else: # one-hot labels
        assert y_batch.shape[1] == num_classes
        assert len(class_names) == num_classes


# -----------------------------
# Logging sanity test
# -----------------------------

def test_logging_helpers_importable():
    from utils import setup_run_logger, setup_child_logger

    run_logger = setup_run_logger("test_run_logger")
    child_logger = setup_child_logger("test_run_logger.child", config.LOGS_DIR / "test.log")

    assert run_logger.name == "test_run_logger"
    assert child_logger.name.startswith("test_run_logger")
