import io
import numpy as np
import pytest
from PIL import Image

import config
from inference import load_saved_model, preprocess_image, predict, InvalidImageError


def _make_image_bytes(size=(50, 50), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(0, 128, 255)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ---------- load_saved_model ----------

def test_load_saved_model_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        load_saved_model("NotARealBackbone", "finetuned")


def test_load_saved_model_rejects_unknown_stage():
    with pytest.raises(ValueError):
        load_saved_model(config.BACKBONE_NAMES[0], "not_a_stage")


# ---------- preprocess_image ----------

def test_preprocess_image_valid_returns_expected_shape():
    arr = preprocess_image(_make_image_bytes())
    assert arr.shape == (1, *config.IMAGE_SIZE, 3)
    assert arr.dtype == np.float32


def test_preprocess_image_is_not_double_normalized():
    """Regression test for the train/serve skew bug: the model applies
    preprocess_input internally, so preprocess_image must return raw
    pixel values (0-255), not backbone-normalized ones."""
    arr = preprocess_image(_make_image_bytes())
    assert arr.max() > 1.0  # would fail if preprocess_input were reapplied here


def test_preprocess_image_invalid_bytes_raises():
    with pytest.raises(InvalidImageError):
        preprocess_image(b"this is not an image")


# ---------- predict ----------

class _FakeModel:
    def __init__(self, logits):
        self._logits = logits

    def predict(self, x, verbose=0):
        return self._logits


def test_predict_returns_expected_top_class():
    class_labels = ["aphid", "beetle", "grasshopper"]
    model = _FakeModel(np.array([[0.1, 5.0, 0.2]]))

    result = predict(model, _make_image_bytes(), class_labels)

    assert result["predicted_class"] == "beetle"
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["class_name"] == "beetle"


def test_predict_top_3_sorted_descending():
    class_labels = ["a", "b", "c", "d"]
    model = _FakeModel(np.array([[1.0, 4.0, 3.0, 2.0]]))

    result = predict(model, _make_image_bytes(), class_labels)
    probs = [entry["probability"] for entry in result["top_3"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_raises_on_invalid_image():
    model = _FakeModel(np.array([[1.0, 1.0, 1.0]]))
    with pytest.raises(InvalidImageError):
        predict(model, b"garbage", ["a", "b", "c"])