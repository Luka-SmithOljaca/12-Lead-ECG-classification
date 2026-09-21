"""Evaluate a selected checkpoint on the held-out PTB-XL fold."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from EDA import get_data_dir, load_metadata
from main import ECGResNet, PTBXLDataset
from metrics import evaluate, print_metrics, select_f1_thresholds, select_screening_thresholds


def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    targets, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            loss_sum += loss_fn(logits, y.to(device)).item() * len(x)
            targets.append(y.numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities), loss_sum / len(dataset)


def error_examples(dataset, targets, probabilities, thresholds):
    """Three most confident false positives and negatives for each label."""
    examples = {}
    for i, label in enumerate(PTBXLDataset.LABELS):
        false_positives = np.where((targets[:, i] == 0) & (probabilities[:, i] >= thresholds[i]))[0]
        false_negatives = np.where((targets[:, i] == 1) & (probabilities[:, i] < thresholds[i]))[0]
        false_positives = false_positives[np.argsort(-probabilities[false_positives, i])[:3]]
        false_negatives = false_negatives[np.argsort(probabilities[false_negatives, i])[:3]]
        def describe(indices):
            return [{"ecg_id": int(dataset.metadata.loc[j, "ecg_id"]),
                     "probability": float(probabilities[j, i])} for j in indices]
        examples[label] = {
            "false_positives": describe(false_positives),
            "false_negatives": describe(false_negatives),
        }
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "Models" / "ecg_resnet_d2_best.pt")
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="Minimum validation recall for MI, STTC, CD and HYP (default: 0.90)")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = get_data_dir(args.data_dir)
    metadata = load_metadata(data_dir)
    val_dataset = PTBXLDataset(metadata[metadata["strat_fold"] == 9], data_dir)
    test_dataset = PTBXLDataset(metadata[metadata["strat_fold"] == 10], data_dir)
    assert (val_dataset.metadata["strat_fold"] == 9).all()
    assert (test_dataset.metadata["strat_fold"] == 10).all()

    device = torch.device("cuda")
    model = ECGResNet(last_block_dilation=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    val_targets, val_probabilities, val_loss = predict(model, val_dataset, device)
    thresholds = select_screening_thresholds(
        val_targets, val_probabilities, PTBXLDataset.LABELS, args.target_recall
    )
    validation = evaluate(val_targets, val_probabilities, PTBXLDataset.LABELS, thresholds)
    test_targets, test_probabilities, test_loss = predict(model, test_dataset, device)
    test = evaluate(test_targets, test_probabilities, PTBXLDataset.LABELS, thresholds)
    f1_thresholds = select_f1_thresholds(val_targets, val_probabilities)
    f1_baseline = evaluate(test_targets, test_probabilities, PTBXLDataset.LABELS, f1_thresholds)

    record = {
        "checkpoint": str(args.checkpoint.resolve().relative_to(project_dir)) if args.checkpoint.resolve().is_relative_to(project_dir) else str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model": "ECGResNet", "last_block_dilation": 2,
        "folds": {"validation": 9, "test": 10},
        "threshold_selection": {
            "abnormal_labels": "highest precision with validation recall at least target, thresholds 0.00 to 1.00 in steps of 0.01",
            "normal_label": "validation F1, thresholds 0.05 to 0.95 in steps of 0.05",
            "target_recall": args.target_recall,
        },
        "validation_loss": val_loss, "validation": validation,
        "test_loss": test_loss, "test": test,
        "f1_threshold_baseline_on_same_checkpoint": f1_baseline,
        "error_examples": error_examples(test_dataset, test_targets, test_probabilities, thresholds),
    }
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    output = results_dir / f"{args.checkpoint.stem}_screening_recall{round(args.target_recall * 100)}_test.json"
    output.write_text(json.dumps(record, indent=2))
    print("Validation (threshold selection):")
    print_metrics(validation)
    print("\nHeld-out test:")
    print_metrics(test)
    print("\nF1 thresholds on the same checkpoint (comparison):")
    print_metrics(f1_baseline)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
