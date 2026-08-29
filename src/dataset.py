import logging

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def create_datasets(logger: logging.Logger | None = None) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, list[str]]:
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
    train_ds = image_dataset_from_directory(TRAIN_DIR, labels="inferred", label_mode="categorical",
                                                        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True,
                                                        seed=SEED)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    if logger:
        logger.info("Classes (%d): %s", num_classes, class_names)


    val_ds = image_dataset_from_directory(VAL_DIR, labels="inferred", label_mode="categorical", class_names=class_names,
                                                      image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                      shuffle=False)  # Not shuffling validation data
    test_ds = image_dataset_from_directory(TEST_DIR, labels="inferred", label_mode="categorical", class_names=class_names,
                                                       image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                       shuffle=False)  # Not shuffling test data

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, num_classes, class_names