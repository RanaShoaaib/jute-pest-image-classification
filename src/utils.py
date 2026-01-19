import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt

def setup_run_logger(name="train", level=logging.INFO):
    """
    Create and return a run logger that logs to console. Does not propagate
    Parameters:
        name: logger name
        level: logging level
    """
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
    logger.propagate = False
    return logger


def setup_child_logger(child_name, log_file, level=logging.INFO):
    """
    Create and return a logger that logs to a file.
    Parameters:
        child_name: logger name (backbone name)
        log_file: path to log file
        level: logging level
    """
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


def flatten_results(results):
    """
    Convert nested results dict into a flat dataframe.
    Expected structure:
    results[backbone][split][metric] = value
    where split in {"train","val","test"} and metric like "precision_macro", ...
    """
    rows = []
    for model_name, splits_dict in results.items():
        row = {"model": model_name}
        for split_name, metrics_dict in splits_dict.items():
            for metric_name, value in metrics_dict.items():
                row[f"{split_name}_{metric_name}"] = value
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df


def plot_history(history, model_name, out_path, logger=None):
    """
    Generates plots of training/val accuracy and loss across epochs.
    Saves plots in directory specified by out_path.
    """
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name}: Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path /"accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path /"loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info("Saved accuracy/loss plots to %s", out_path)


def evaluate_on_dataset(model, dataset, set_name, num_classes, class_names, out_path, logger=None):
    """
    Computes macro Precision, Recall, F1, and AUC on a dataset.
    Saves a detailed classification report in out_path and returns metrics dict.
    """
    y_prob = model.predict(dataset, verbose=0)
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