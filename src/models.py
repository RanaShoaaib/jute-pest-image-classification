from config import LEARNING_RATE, L2_LAMBDA, DROPOUT_RATE
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def get_backbone(backbone_name, input_shape):
    """
    Returns a pretrained ImageNet backbone (without top).
    All layers are frozen (non-trainable).
    """
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

def build_head(x, num_classes):
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(L2_LAMBDA))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return layers.Dense(num_classes, activation="softmax")(x)


def get_preprocess_fn(backbone_name):
    if backbone_name == "EfficientNetB0":
        return keras.applications.efficientnet.preprocess_input
    elif backbone_name == "VGG16":
        return keras.applications.vgg16.preprocess_input
    elif backbone_name == "DenseNet201":
        return keras.applications.densenet.preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def build_transfer_model(backbone_name, input_shape, num_classes):
    base_model = get_backbone(backbone_name, input_shape)

    inputs = keras.Input(shape=input_shape)
    preprocess = get_preprocess_fn(backbone_name)
    x = preprocess(inputs)
    x = base_model(x, training=False)

    outputs = build_head(x, num_classes)

    model = keras.Model(inputs, outputs, name=f"{backbone_name}_jute_pests")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])

    return model
