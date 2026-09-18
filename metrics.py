# metrics.py
from sklearn.metrics import roc_auc_score


def per_class_auroc(targets, probabilities, labels):
    scores = {}

    for i, label in enumerate(labels):
        scores[label] = roc_auc_score(
            targets[:, i],
            probabilities[:, i],
        )

    return scores