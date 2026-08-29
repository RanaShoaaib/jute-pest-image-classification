from config import L2_LAMBDA, DROPOUT_RATE, BACKBONE_NAMES
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Sequential
from collections.abc import Callable

def data_augmenter()-> Sequential:
    """Create the data augmentation pipeline used during training."""
    data_augmentation = Sequential([
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.05),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ], name="data_augmentation")
    return data_augmentation


def get_backbone(backbone_name: str, input_shape: tuple[int, int, int]) -> keras.Model:
    """Return a frozen pretrained ImageNet backbone without its classifier."""
    if backbone_name == "EfficientNetB0":
        base_model = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif backbone_name == "VGG16":
        base_model = keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    elif backbone_name == "DenseNet201":
        base_model = keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base_model.trainable = False  # freeze all layers
    return base_model

def build_head(x: keras.KerasTensor, num_classes: int) -> keras.KerasTensor:
    """Build the classification head on top of backbone features."""
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(L2_LAMBDA))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return layers.Dense(num_classes)(x)


def get_preprocess_fn(backbone_name: str) -> Callable:
    """Return the preprocessing function for the specified backbone."""
    if backbone_name == "EfficientNetB0":
        return keras.applications.efficientnet.preprocess_input
    elif backbone_name == "VGG16":
        return keras.applications.vgg16.preprocess_input
    elif backbone_name == "DenseNet201":
        return keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def build_transfer_model(backbone_name: str, input_shape: tuple[int,int,int], num_classes:int) -> keras.Model:
    """Build a transfer-learning model for the specified backbone."""
    inputs = keras.Input(shape=input_shape)
    data_augmentation = data_augmenter()
    x = data_augmentation(inputs)
    preprocess = get_preprocess_fn(backbone_name)
    x = preprocess(x)
    base_model = get_backbone(backbone_name, input_shape)
    x = base_model(x, training=False)
    outputs = build_head(x, num_classes)

    model = keras.Model(inputs, outputs, name=f"{backbone_name}_jute_pests")
    return model

def unfreeze_base_layers(model: keras.Model, backbone: str, frac: float = 0.2) -> keras.Model:
    """Unfreeze the final fraction of backbone layers for fine-tuning."""
    if not isinstance(backbone,str) or backbone not in BACKBONE_NAMES:
        raise ValueError(f"Invalid backbone: {backbone}")

    layer_name = None
    for layer in model.layers:
        if layer.name.lower() == backbone.lower():
            layer_name = layer.name
            break
    if not layer_name:
        raise ValueError(f"Layer {backbone} is not in the provided model")

    base_model = model.get_layer(layer_name)
    base_model.trainable = True
    split_index = int(len(base_model.layers) * (1-frac))
    for i, layer in enumerate(base_model.layers):
        if i >= split_index:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = False
    return model