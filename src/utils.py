import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import shutil
from typing import Any
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from config import LOGS_DIR, MODELS_DIR, ARTIFACT_DIR

def setup_run_logger(name: str = "train", level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Create a run logger that writes to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is None:
        log_file = LOGS_DIR / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def setup_child_logger(child_name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Create a child logger that writes to a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(child_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.propagate = True
    return logger


def combine_histories(initial_hist: tf.keras.callbacks.History, finetune_hist: tf.keras.callbacks.History) -> dict:
    """Combine initial and fine-tuning training histories."""
    combined_hist = {}
    for key in initial_hist.history:
        combined_hist[key] = initial_hist.history[key] + finetune_hist.history[key]
    return combined_hist


def flatten_results(results: dict[str,Any]) -> pd.DataFrame:
    """Flatten nested model evaluation results into a DataFrame."""
    rows = []
    for model_name, splits_dict in results.items():
        row = {"model": model_name}
        for split_name, metrics_dict in splits_dict.items():
            for metric_name, value in metrics_dict.items():
                row[f"{split_name}_{metric_name}"] = value
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df


def plot_history(history: dict[str, Any], model_name: str, initial_epochs_trained: int, out_path: Path, logger: logging.Logger = None) -> None:
    """
    Generates plots of training/val accuracy and loss across epochs.
    Saves plots in directory specified by out_path.
    """
    # Accuracy
    plt.figure()
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.plot([initial_epochs_trained - 1, initial_epochs_trained - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name}: Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path /"accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot([initial_epochs_trained - 1, initial_epochs_trained - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path /"loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info("Saved accuracy/loss plots to %s", out_path)


def evaluate_on_dataset(model: tf.keras.Model, dataset: tf.data.Dataset, set_name: str, class_names: list[str], out_path: Path, logger: logging.Logger=None) -> dict[str, float]:
    """
    Computes macro Precision, Recall, F1, and AUC on a dataset.
    Saves a detailed classification report in out_path and returns metrics dict.
    """
    y_logits = model.predict(dataset, verbose=0)
    y_prob = tf.nn.softmax(y_logits).numpy()
    y_pred = np.argmax(y_prob, axis=1)

    # True labels (batched, one-hot)
    y_true_one_hot = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=1)

    clf_rep = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(out_path / f"{set_name}_classification_report.txt", "w") as f:
        f.write(clf_rep)

    if logger:
        logger.info("Saved %s classification report to %s", set_name, out_path)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    # multi-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan  # In case of numerical issues, e.g., only one class present in y_true

    metrics = {"precision_macro": precision, "recall_macro": recall, "f1_macro": f1, "auc_macro": auc}

    return metrics

def generate_manifest(df: pd.DataFrame, best_stage_dict: dict[str, str], num_classes: int, class_names: list[str]) -> dict[str, Any]:
    """Create and save metadata for the selected deployment model."""
    best_backbone = df["val_auc_macro"].idxmax()
    best_row = df.loc[best_backbone]
    best_stage = best_stage_dict[best_backbone]

    manifest = {
        "selected_backbone": best_backbone,
        "stage": best_stage,
        "val_auc_macro": float(best_row["val_auc_macro"]),
        "test_auc_macro": float(best_row["test_auc_macro"]),
        "num_classes": num_classes,
        "class_names": class_names,
    }

    # Full experimental record — stays in models/, gitignored
    with open(MODELS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    # Deployment copy — manifest + matching model checkpoint together
    with open(ARTIFACT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    return manifest

def copy_best_model_for_service(manifest: dict[str, Any]) -> None:
    """Copy the manifest-selected model checkpoint into the artifact directory."""
    backbone = manifest["selected_backbone"]
    stage = manifest["stage"]
    filename = f"best_{backbone}_{stage}.keras"

    src_path = MODELS_DIR / filename
    dst_path = ARTIFACT_DIR / filename

    if not src_path.exists():
        raise FileNotFoundError(f"Expected model checkpoint not found: {src_path}")

    for stale_file in ARTIFACT_DIR.glob("best_*.keras"):
        stale_file.unlink()

    shutil.copy2(src_path, dst_path)