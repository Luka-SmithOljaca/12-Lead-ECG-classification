import argparse
import json
import random
from pathlib import Path
import torch
import wfdb
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from metrics import evaluate, print_metrics, select_screening_thresholds
from EDA import LABELS, get_data_dir, load_metadata

class PTBXLDataset(torch.utils.data.Dataset):
    LABELS = LABELS

    def __init__(self, metadata, data_dir):
        self.metadata = metadata.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.signals = torch.empty(
            (len(self.metadata), 12, 1000), dtype=torch.float32
        )
        self.targets = torch.from_numpy(
            self.metadata[self.LABELS].to_numpy(dtype=np.float32, copy=True)
        )

        for index, row in self.metadata.iterrows():
            record_path = self.data_dir / row["filename_lr"]
            signal, _ = wfdb.rdsamp(str(record_path))
            self.signals[index] = torch.from_numpy(signal.T)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.signals[index], self.targets[index]

class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.AdaptiveAvgPool1d(output_size=1),
        )

        self.classifier = nn.Linear(in_features=64, out_features=5)

    def forward(self, x):
        x = self.features(x)   # (batch, 64, 1)
        x = x.squeeze(-1)      # (batch, 64)
        return self.classifier(x)  # (batch, 5)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = 2 * dilation  # Preserves length for a size-5 kernel.

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=5,
            padding=padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5,
            padding=padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Addition requires both paths to have the same number of channels.
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)

class ECGResNet(nn.Module):
    def __init__(self, last_block_dilation=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            nn.MaxPool1d(2),
            ResidualBlock(32, 64),
            nn.MaxPool1d(2),
            ResidualBlock(64, 64, dilation=last_block_dilation),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, 5)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="PTB-XL directory; defaults to Data/... or PTBXL_DATA_DIR")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="Minimum validation recall for MI, STTC, CD and HYP (default: 0.90)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = get_data_dir(args.data_dir)
    metadata = load_metadata(data_dir)
    train_dataset = PTBXLDataset(metadata[metadata["strat_fold"].between(1, 8)], data_dir)
    val_dataset = PTBXLDataset(metadata[metadata["strat_fold"] == 9], data_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_block_dilation = 2
    model = ECGResNet(last_block_dilation=last_block_dilation).to(device)
    positive_counts = train_dataset.targets.sum(dim=0)
    negative_counts = len(train_dataset) - positive_counts
    if torch.any(positive_counts == 0):
        missing_labels = [
            label
            for label, count in zip(PTBXLDataset.LABELS, positive_counts)
            if count == 0
        ]
        raise ValueError(
            f"Cannot calculate positive class weights; no training examples for: "
            f"{', '.join(missing_labels)}"
        )
    pos_weight = (negative_counts / positive_counts).to(device)
    train_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    evaluation_loss_fn = nn.BCEWithLogitsLoss()
    class_weights = {
        label: float(weight)
        for label, weight in zip(PTBXLDataset.LABELS, pos_weight.cpu())
    }
    print(f"Positive class weights: {class_weights}")
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    patience = 8
    project_dir = Path(__file__).resolve().parent
    models_dir = project_dir / "Models"
    results_dir = project_dir / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    checkpoint = models_dir / f"ecg_resnet_d{last_block_dilation}_weighted_bce_best.pt"
    best_auroc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        # Training: calculate gradients and update weights
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = train_loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_dataset)

        # Validation: measure performance without updating weights
        model.eval()
        val_loss_sum = 0.0

        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = evaluation_loss_fn(logits, y)
                val_loss_sum += loss.item() * x.size(0)

                all_targets.append(y.cpu().numpy())
                all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

        val_loss = val_loss_sum / len(val_dataset)

        targets = np.concatenate(all_targets)
        probabilities = np.concatenate(all_probabilities)
        metrics = evaluate(targets, probabilities, PTBXLDataset.LABELS)
        macro_auroc = metrics["macro_auroc"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "validation_loss": val_loss, "validation_macro_auroc": macro_auroc})

        print(
            f"Epoch {epoch + 1}: "
            f"train loss = {train_loss:.4f}, "
            f"val loss = {val_loss:.4f}, "
            f"val macro AUROC = {macro_auroc:.4f}"
        )
        if macro_auroc > best_auroc:
            best_auroc = macro_auroc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
            print(" Saved new best checkpoint")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Stopped early after {epoch + 1} epochs")
            break

    # Re-evaluate the selected checkpoint, since the final epoch may not be the best.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    all_targets, all_probabilities = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            all_targets.append(y.numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    thresholds = select_screening_thresholds(
        targets, probabilities, PTBXLDataset.LABELS, args.target_recall
    )
    validation = evaluate(targets, probabilities, PTBXLDataset.LABELS, thresholds)
    record = {
        "model": "ECGResNet", "last_block_dilation": last_block_dilation,
        "seed": args.seed, "learning_rate": learning_rate, "batch_size": 32,
        "loss": "weighted_BCEWithLogitsLoss",
        "positive_class_weights": class_weights,
        "max_epochs": args.epochs, "patience": patience,
        "best_epoch": best_epoch, "checkpoint": str(checkpoint.relative_to(project_dir)),
        "folds": {"train": "1-8", "validation": 9, "test": 10},
        "threshold_selection": {
            "abnormal_labels": "highest precision with validation recall at least target, thresholds 0.00 to 1.00 in steps of 0.01",
            "normal_label": "validation F1, thresholds 0.05 to 0.95 in steps of 0.05",
            "target_recall": args.target_recall,
        },
        "validation": validation, "history": history,
    }
    result_path = results_dir / f"ecg_resnet_d{last_block_dilation}_weighted_bce_screening_validation.json"
    result_path.write_text(json.dumps(record, indent=2))
    print(f"Best epoch: {best_epoch}")
    print_metrics(validation)


if __name__ == "__main__":
    main()
