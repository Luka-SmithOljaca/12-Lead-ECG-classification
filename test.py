from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from main import ECGResNet, PTBXLDataset, test_dataset
from metrics import per_class_auroc


def main():
    assert (test_dataset.metadata["strat_fold"] == 10).all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(__file__).with_name("ecg_resnet_d2_best.pt")

    model = ECGResNet(last_block_dilation=2).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * x.size(0)
            all_targets.append(y.cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    scores = per_class_auroc(targets, probabilities, PTBXLDataset.LABELS)
    macro_auroc = sum(scores.values()) / len(scores)

    print(f"Test ECGs: {len(test_dataset)}")
    print(f"Test loss: {loss_sum / len(test_dataset):.4f}")
    print(f"Test macro AUROC: {macro_auroc:.4f}")
    for label, score in scores.items():
        print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
