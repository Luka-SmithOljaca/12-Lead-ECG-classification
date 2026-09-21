"""Metrics for five independent PTB-XL diagnostic labels."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def evaluate(targets, probabilities, labels, thresholds=None):
    """Return JSON-friendly macro and per-class metrics at fixed thresholds."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    thresholds = np.full(len(labels), 0.5) if thresholds is None else np.asarray(thresholds)
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError("Targets and probabilities must have one column per label")
    if thresholds.shape != (len(labels),):
        raise ValueError("Provide one threshold per label")

    predictions = probabilities >= thresholds
    per_class = {}
    for i, label in enumerate(labels):
        actual = targets[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predictions[:, i], average="binary", zero_division=0
        )
        auroc = float(roc_auc_score(actual, probabilities[:, i])) if len(np.unique(actual)) == 2 else None
        per_class[label] = {
            "positives": int(actual.sum()),
            "predicted_positives": int(predictions[:, i].sum()),
            "false_positives": int(((actual == 0) & predictions[:, i]).sum()),
            "false_negatives": int(((actual == 1) & ~predictions[:, i]).sum()),
            "prevalence": float(actual.mean()),
            "auroc": auroc,
            "average_precision": float(average_precision_score(actual, probabilities[:, i])),
            "threshold": float(thresholds[i]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def macro(key):
        values = [row[key] for row in per_class.values() if row[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "samples": int(len(targets)),
        "macro_auroc": macro("auroc"),
        "macro_average_precision": macro("average_precision"),
        "macro_f1": macro("f1"),
        "per_class": per_class,
    }


def select_f1_thresholds(targets, probabilities):
    """Choose per-class F1 thresholds on validation predictions only."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    grid = np.arange(0.05, 0.96, 0.05)
    chosen = []
    for i in range(targets.shape[1]):
        scores = [precision_recall_fscore_support(
            targets[:, i], probabilities[:, i] >= threshold,
            average="binary", zero_division=0
        )[2] for threshold in grid]
        chosen.append(float(grid[int(np.argmax(scores))]))
    return chosen


def select_screening_thresholds(targets, probabilities, labels, target_recall=0.90):
    """Meet validation recall for abnormal labels; keep NORM's F1 threshold."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be between 0 and 1")
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError("Targets and probabilities must have one column per label")

    thresholds = select_f1_thresholds(targets, probabilities)
    grid = np.linspace(0, 1, 101)
    for i, label in enumerate(labels):
        if label == "NORM":
            continue
        if targets[:, i].sum() == 0:
            raise ValueError(f"No positive validation cases for {label}")
        choices = []
        for threshold in grid:
            predicted = probabilities[:, i] >= threshold
            true_positives = np.count_nonzero(predicted & (targets[:, i] == 1))
            recall = true_positives / targets[:, i].sum()
            if recall >= target_recall:
                precision = true_positives / predicted.sum() if predicted.any() else 0.0
                choices.append((precision, float(threshold)))
        # If precision ties, prefer the higher threshold and fewer alerts.
        thresholds[i] = max(choices)[1]
    return thresholds


def print_metrics(metrics):
    print(f"Samples: {metrics['samples']}")
    for key in ("macro_auroc", "macro_average_precision", "macro_f1"):
        value = metrics[key]
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: n/a")
    print("Class  Prev.  AUROC  AP     Thr.  Prec.  Recall  F1     FP    FN")
    for label, row in metrics["per_class"].items():
        auc = "n/a" if row["auroc"] is None else f"{row['auroc']:.3f}"
        print(f"{label:5}  {row['prevalence']:.3f}  {auc:5}  "
              f"{row['average_precision']:.3f}  {row['threshold']:.2f}  "
              f"{row['precision']:.3f}  {row['recall']:.3f}   {row['f1']:.3f}  "
              f"{row['false_positives']:4d}  {row['false_negatives']:4d}")
