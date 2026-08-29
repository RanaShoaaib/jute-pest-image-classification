import json
import numpy as np, tensorflow as tf
from tensorflow import keras
from config import BACKBONE_NAMES, MODELS_DIR, RESULTS_DIR, LOGS_DIR, INPUT_SHAPE, PATIENCE, LR_PATIENCE, EPOCHS, \
    FINETUNE_EPOCHS, SEED, LEARNING_RATE, FINETUNE_LR, LR_REDUCTION_FACTOR, UNFREEZE_LAYERS_FRAC, ARTIFACT_DIR
from dataset import create_datasets
from models import build_transfer_model, unfreeze_base_layers
from utils import plot_history, evaluate_on_dataset, flatten_results, setup_run_logger, setup_child_logger, combine_histories, generate_manifest, copy_best_model_for_service


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    run_logger = setup_run_logger(name="train")
    run_logger.info("Starting training run for %d backbones: %s", len(BACKBONE_NAMES), BACKBONE_NAMES)
    train_ds, val_ds, test_ds, num_classes, class_names = create_datasets(run_logger)
    results = {}
    best_stage_for_backbone = {}

    for backbone in BACKBONE_NAMES:
        logger = setup_child_logger(child_name=f"train.{backbone}", log_file=LOGS_DIR / f"{backbone}.log")
        logger.info("Starting training for %s", backbone)

        model = build_transfer_model(backbone, INPUT_SHAPE, num_classes)
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        old = logger.propagate
        logger.propagate = False
        model.summary(print_fn=logger.info)
        logger.propagate = old

        callbacks_initial = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=MODELS_DIR/ f"best_{backbone}_initial.keras", monitor="val_loss",
                                                     save_best_only=True, save_weights_only=False),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_REDUCTION_FACTOR, patience=LR_PATIENCE, min_lr=1e-6)
        ]

        callbacks_finetune = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=MODELS_DIR / f"best_{backbone}_finetuned.keras", monitor="val_loss",
                                            save_best_only=True, save_weights_only=False),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_REDUCTION_FACTOR, patience=LR_PATIENCE,
                                              min_lr=1e-8)
        ]

        history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks_initial, verbose=1)

        initial_epochs_trained = len(history.history['loss'])
        total_epochs = initial_epochs_trained + FINETUNE_EPOCHS

        model = unfreeze_base_layers(model, backbone, frac=UNFREEZE_LAYERS_FRAC)
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=FINETUNE_LR),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        logger.info(f"Starting finetuning for {backbone}")
        history_fine = model.fit(train_ds, epochs=total_epochs, validation_data=val_ds, initial_epoch=initial_epochs_trained, callbacks=callbacks_finetune, verbose=1)

        initial_best_loss = min(history.history["val_loss"])
        finetune_best_loss = min(history_fine.history["val_loss"])

        if finetune_best_loss < initial_best_loss:
            model = keras.models.load_model(MODELS_DIR / f"best_{backbone}_finetuned.keras")
            best_stage_for_backbone[backbone] = "finetuned"
            logger.info(
                "Fine-tuned model selected: val_loss %.4f vs initial %.4f",
                finetune_best_loss,
                initial_best_loss,
            )
        else:
            model = keras.models.load_model(MODELS_DIR / f"best_{backbone}_initial.keras")
            best_stage_for_backbone[backbone] = "initial"
            logger.info(
                "Initial model retained: val_loss %.4f vs fine-tuned %.4f",
                initial_best_loss,
                finetune_best_loss,
            )

        combined_history = combine_histories(history, history_fine)
        out_path = RESULTS_DIR / backbone
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "history.json", "w") as f:
            json.dump(combined_history, f, indent=2)
        plot_history(combined_history, backbone, initial_epochs_trained, out_path, logger)

        # Evaluate on val, test
        val_metrics = evaluate_on_dataset(model, val_ds, f"{backbone}_Val", class_names, out_path, logger)
        test_metrics = evaluate_on_dataset(model, test_ds, f"{backbone}_Test", class_names, out_path, logger)

        results[backbone] = {"val": val_metrics, "test": test_metrics}

    df = flatten_results(results)
    run_logger.info("\n\t\t\t==============Summary of Results==============\n%s", df.to_string())

    df.to_csv(RESULTS_DIR / "models_comparison.csv") # Save results as CSV
    with open(RESULTS_DIR / "models_comparison.json", "w") as f: # Save results as JSON
        json.dump(results, f, indent=2)
    run_logger.info("Performance metrics for models saved to %s",RESULTS_DIR)
    manifest = generate_manifest(df, best_stage_for_backbone, num_classes, class_names)
    run_logger.info("Manifest for best model saved to %s", MODELS_DIR)
    run_logger.info("Manifest for best model saved to %s", ARTIFACT_DIR)
    copy_best_model_for_service(manifest)
    run_logger.info("Best model saved to %s", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
