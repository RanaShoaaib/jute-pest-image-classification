import json
import numpy as np, tensorflow as tf
from tensorflow import keras
from config import BACKBONE_NAMES, MODELS_DIR, RESULTS_DIR, LOGS_DIR, INPUT_SHAPE, PATIENCE, EPOCHS, SEED
from dataset import create_datasets
from models import build_transfer_model
from utils import plot_history, evaluate_on_dataset, flatten_results, setup_run_logger, setup_child_logger


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    run_logger = setup_run_logger(name="train")
    run_logger.info("Starting training run for %d backbones: %s", len(BACKBONE_NAMES), BACKBONE_NAMES)
    train_ds, val_ds, test_ds, num_classes, class_names = create_datasets(run_logger)
    results = {}

    for backbone in BACKBONE_NAMES:
        logger = setup_child_logger(child_name=f"train.{backbone}", log_file=LOGS_DIR / f"{backbone}.log")
        logger.info("Starting training for %s", backbone)

        model = build_transfer_model(backbone, INPUT_SHAPE, num_classes)
        old = logger.propagate
        logger.propagate = False
        model.summary(print_fn=logger.info)
        logger.propagate = old

        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                     keras.callbacks.ModelCheckpoint(filepath=MODELS_DIR/ f"best_{backbone}.keras", monitor="val_loss",
                                                     save_best_only=True, save_weights_only=False)]

        history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)
        out_path = RESULTS_DIR / backbone
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "history.json", "w") as f:
            json.dump(history.history, f, indent=2)
        plot_history(history, backbone, out_path, logger)

        # Evaluate on train, val, test
        train_metrics = evaluate_on_dataset(model, train_ds, f"{backbone}_Train", num_classes, class_names, out_path, logger)
        val_metrics = evaluate_on_dataset(model, val_ds, f"{backbone}_Val", num_classes, class_names, out_path, logger)
        test_metrics = evaluate_on_dataset(model, test_ds, f"{backbone}_Test", num_classes, class_names, out_path, logger)

        results[backbone] = {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    df = flatten_results(results)
    run_logger.info("\n\t\t\t==============Summary of Results==============\n%s", df.to_string())

    df.to_csv(RESULTS_DIR / "models_comparison.csv") # Save results as CSV
    with open(RESULTS_DIR / "models_comparison.json", "w") as f: # Save results as JSON
        json.dump(results, f, indent=2)
    run_logger.info("Performance metrics for models saved to %s",RESULTS_DIR)


if __name__ == "__main__":
    main()
