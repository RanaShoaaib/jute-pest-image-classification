import numpy as np
from config import BACKBONE_NAMES, IMAGE_SIZE, ARTIFACT_DIR
import io
from typing import Any
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import tensorflow.keras as keras


VALID_STAGES: set[str] = {'finetuned','initial'}


class InvalidImageError(Exception):
    pass

def load_saved_model(backbone_name:str, stage: str) -> keras.models.Model:
    """Load the best performing saved model."""
    if backbone_name not in BACKBONE_NAMES or stage not in VALID_STAGES:
        raise ValueError("Invalid Arguments.")

    filepath = ARTIFACT_DIR/f"best_{backbone_name}_{stage}.keras"
    model = keras.models.load_model(filepath)
    return model

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes and return an RGB model input."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            arr = np.array(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Uploaded file is not a valid image") from exc
    return np.expand_dims(arr, axis=0)

def predict(model: keras.models.Model, image_bytes: bytes, class_labels: list[str]) -> dict[str, Any]:
    """Run inference and return the predicted class and top probabilities."""
    x = preprocess_image(image_bytes)
    logits = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
    probs = tf.nn.softmax(logits).numpy()

    top_indices = np.argsort(probs)[::-1][:3]
    top_3 = [{"class_name": class_labels[i], "probability": float(probs[i])} for i in top_indices]

    return {
        "predicted_class": class_labels[int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "top_3": top_3,
    }