from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CACHE_DIR, IMAGE_SIZE, BATCH_SIZE, SEED

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


def create_datasets(logger=None):
    """
    Creates training, validation, and test tf.data.Datasets.
    Assumes directory structure:
        DATA_ROOT/
            train/class1/...
            val/class1/...
            test/class1/...
    """

    # Load data using image_dataset_from_directory. Note: labels inferred from folder structure and resizing done.
    if logger:
        logger.info("Loading datasets: train=%s val=%s test=%s", TRAIN_DIR, VAL_DIR, TEST_DIR)
    train_ds = keras.utils.image_dataset_from_directory(TRAIN_DIR, labels="inferred", label_mode="categorical",
                                                        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True,
                                                        seed=SEED)
    val_ds = keras.utils.image_dataset_from_directory(VAL_DIR, labels="inferred", label_mode="categorical",
                                                      image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                      shuffle=False)  # Not shuffling validation data
    test_ds = keras.utils.image_dataset_from_directory(TEST_DIR, labels="inferred", label_mode="categorical",
                                                       image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                       shuffle=False)  # Not shuffling test data
    class_names = train_ds.class_names
    num_classes = len(class_names)
    if logger:
        logger.info("Classes (%d): %s", num_classes, class_names)

    # Define Augmentation and Preprocessing Layers
    data_augmentation_layers = Sequential([
        layers.Resizing(256, 256),
        layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ], name="data_augmentation")

    # Apply layers to the Datasets
    AUTOTUNE = tf.data.AUTOTUNE
    def prepare_dataset(ds, cache_name, augment=False, cache_mode="none"):
        if cache_mode == "disk":
            ds = ds.cache(str(CACHE_DIR/f"{cache_name}.cache"))
        elif cache_mode == "memory":
            ds = ds.cache()

        if augment:  # Apply augmentation only to training data
            ds = ds.map(lambda x, y: (data_augmentation_layers(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
        return ds.prefetch(buffer_size=AUTOTUNE)  # Prefetch for performance

    train_ds = prepare_dataset(train_ds, "train",augment=True, cache_mode="disk")
    val_ds = prepare_dataset(val_ds, "val", cache_mode="none")
    test_ds = prepare_dataset(test_ds, "test", cache_mode="none")

    return train_ds, val_ds, test_ds, num_classes, class_names