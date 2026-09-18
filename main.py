from pathlib import Path
import torch
import wfdb
import numpy as np
from metrics import per_class_auroc
from EDA import metadata, data_dir

class PTBXLDataset(torch.utils.data.Dataset):
    LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

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

train_metadata = metadata[metadata["strat_fold"].between(1, 8)]
val_metadata = metadata[metadata["strat_fold"] == 9]
test_metadata = metadata[metadata["strat_fold"] == 10]

train_dataset = PTBXLDataset(train_metadata, data_dir)
val_dataset = PTBXLDataset(val_metadata, data_dir)
test_dataset = PTBXLDataset(test_metadata, data_dir)

import torch
from torch import nn
from torch.utils.data import DataLoader


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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda")
last_block_dilation = 2
model = ECGResNet(last_block_dilation=last_block_dilation).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    x, y = train_dataset[0]
    print(x.shape, y.shape)

    max_epochs = 60
    patience = 8
    best_auroc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training: calculate gradients and update weights
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
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
                loss = loss_fn(logits, y)
                val_loss_sum += loss.item() * x.size(0)

                all_targets.append(y.cpu().numpy())
                all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

        val_loss = val_loss_sum / len(val_dataset)

        targets = np.concatenate(all_targets)
        probabilities = np.concatenate(all_probabilities)
        scores = per_class_auroc(targets, probabilities, PTBXLDataset.LABELS)
        macro_auroc = sum(scores.values()) / len(scores)

        print(
            f"Epoch {epoch + 1}: "
            f"train loss = {train_loss:.4f}, "
            f"val loss = {val_loss:.4f}, "
            f"val macro AUROC = {macro_auroc:.4f}"
        )
        for label, score in scores.items():
            print(f"  {label}: {score:.3f}")

        if macro_auroc > best_auroc:
            best_auroc = macro_auroc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                f"ecg_resnet_d{last_block_dilation}_best.pt",
            )
            print(" Saved new best checkpoint")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Stopped early after {epoch + 1} epochs")
            break

    print(f"Best epoch: {best_epoch}, val macro AUROC: {best_auroc:.4f}")
