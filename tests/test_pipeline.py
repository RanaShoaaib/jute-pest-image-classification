import pytest
import tensorflow as tf
import config
from models import build_transfer_model, get_preprocess_fn, unfreeze_base_layers
from utils import combine_histories, flatten_results

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


@pytest.mark.parametrize("backbone", config.BACKBONE_NAMES)
def test_unfreeze_base_layers(backbone):
    num_classes = 3
    # Build model with frozen backbone
    model = build_transfer_model(
        backbone_name=backbone,
        input_shape=config.INPUT_SHAPE,
        num_classes=num_classes
    )
    base_model = next(
        layer
        for layer in model.layers
        if layer.name.lower() == backbone.lower()
    )
    # Backbone should initially be completely frozen
    assert base_model.trainable is False
    assert all(not layer.trainable for layer in base_model.layers)
    # Unfreeze final 25% of backbone
    model = unfreeze_base_layers(model, backbone, frac=0.25)
    base_model = next(
        layer
        for layer in model.layers
        if layer.name.lower() == backbone.lower()
    )
    assert base_model.trainable is True
    split_index = int(len(base_model.layers) * 0.75)
    # First 75% should remain frozen
    for layer in base_model.layers[:split_index]:
        assert layer.trainable is False
    # Final 25% should be trainable except BatchNormalization layers
    for layer in base_model.layers[split_index:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False
        else:
            assert layer.trainable is True


# -----------------------------
# Dataset tests
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
# utils.py logic tests
# -----------------------------

class _FakeHistory:
    """Minimal stand-in for keras.callbacks.History — combine_histories only reads .history."""
    def __init__(self, history_dict):
        self.history = history_dict

def test_combine_histories_concatenates_per_key():
    initial = _FakeHistory({
        "loss": [1.0, 0.8], "val_loss": [1.1, 0.9],
        "accuracy": [0.5, 0.6], "val_accuracy": [0.4, 0.55],
    })
    finetune = _FakeHistory({
        "loss": [0.7, 0.6], "val_loss": [0.85, 0.8],
        "accuracy": [0.65, 0.7], "val_accuracy": [0.6, 0.65],
    })

    combined = combine_histories(initial, finetune)

    assert combined["loss"] == [1.0, 0.8, 0.7, 0.6]
    assert combined["val_loss"] == [1.1, 0.9, 0.85, 0.8]
    assert combined["accuracy"] == [0.5, 0.6, 0.65, 0.7]
    assert combined["val_accuracy"] == [0.4, 0.55, 0.6, 0.65]
    assert set(combined.keys()) == set(initial.history.keys())


def test_flatten_results_produces_expected_columns_and_index():
    results = {
        "EfficientNetB0": {
            "val": {"precision_macro": 0.80, "recall_macro": 0.78, "f1_macro": 0.79, "auc_macro": 0.91},
            "test": {"precision_macro": 0.77, "recall_macro": 0.75, "f1_macro": 0.76, "auc_macro": 0.89},
        },
        "VGG16": {
            "val": {"precision_macro": 0.70, "recall_macro": 0.68, "f1_macro": 0.69, "auc_macro": 0.85},
            "test": {"precision_macro": 0.66, "recall_macro": 0.65, "f1_macro": 0.655, "auc_macro": 0.83},
        },
    }

    df = flatten_results(results)

    assert list(df.index) == ["EfficientNetB0", "VGG16"]
    assert df.index.name == "model"
    expected_cols = {
        "val_precision_macro", "val_recall_macro", "val_f1_macro", "val_auc_macro",
        "test_precision_macro", "test_recall_macro", "test_f1_macro", "test_auc_macro",
    }
    assert expected_cols == set(df.columns)
    assert df.loc["EfficientNetB0", "val_f1_macro"] == 0.79
    assert df.loc["VGG16", "test_auc_macro"] == 0.83


# -----------------------------
# Logging sanity test
# -----------------------------

def test_logging_helpers_importable():
    from utils import setup_run_logger, setup_child_logger

    run_logger = setup_run_logger("test_run_logger")
    child_logger = setup_child_logger("test_run_logger.child", config.LOGS_DIR / "test.log")

    assert run_logger.name == "test_run_logger"
    assert child_logger.name.startswith("test_run_logger")
