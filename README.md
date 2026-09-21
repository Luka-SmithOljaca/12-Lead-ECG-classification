# PTB-XL ECG classification

This project predicts five diagnostic superclasses from 12-lead ECGs sampled at 100 Hz:

- `NORM` — normal ECG
- `MI` — myocardial infarction
- `STTC` — ST/T change
- `CD` — conduction disturbance
- `HYP` — hypertrophy

This is a multilabel problem, so a recording can belong to more than one class. The project uses the official PTB-XL folds: folds 1–8 for training, fold 9 for validation, and fold 10 as the held-out test set.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then install the project dependencies:

```powershell
uv sync
```

Download PTB-XL 1.0.3 and extract it to:

```text
Data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
```

The folder should contain `ptbxl_database.csv`. If the dataset is stored elsewhere, either set `PTBXL_DATA_DIR` or pass its location with `--data-dir`.

## Training

Run the training script from the project directory:

```powershell
uv run python main.py
```

The current model is a small 1D ResNet. It trains for up to 60 epochs and stops early if validation macro AUROC does not improve for eight epochs. The best checkpoint is selected by validation macro AUROC.

Training now uses weighted binary cross-entropy to reduce the effect of class imbalance. A separate positive weight is calculated for each diagnosis using the training folds only:

```text
positive weight = number of negative examples / number of positive examples
```

The weights are printed at the start of training and written to the result file. Training loss is weighted, while validation loss remains ordinary binary cross-entropy so that it is easier to compare between runs.

The weighted run writes:

```text
Models/ecg_resnet_d2_weighted_bce_best.pt
results/ecg_resnet_d2_weighted_bce_screening_validation.json
```

The weighted filenames are separate from the original checkpoint and results, so the unweighted baseline is not overwritten.

Useful training options are:

```powershell
uv run python main.py --epochs 40 --seed 123
uv run python main.py --data-dir C:/path/to/ptb-xl
```

Run `uv run python main.py --help` to see every option.

## Evaluation

Evaluate the weighted checkpoint with:

```powershell
uv run python test.py --checkpoint Models/ecg_resnet_d2_weighted_bce_best.pt
```

The evaluator chooses thresholds using validation fold 9 and then applies those fixed thresholds to test fold 10. The test fold is not used to choose the model or thresholds.

Two threshold policies are reported:

- **F1 thresholds:** the best validation F1 threshold for each class, searched from 0.05 to 0.95 in steps of 0.05. This gives the best balance of precision and recall in the current results and is the preferred general-purpose setting.
- **Screening thresholds:** for `MI`, `STTC`, `CD`, and `HYP`, choose the highest-precision validation threshold that reaches at least 90% recall. `NORM` still uses its F1 threshold. This setting favours sensitivity and produces more false positives.

The screening recall target can be changed when needed:

```powershell
uv run python test.py --checkpoint Models/ecg_resnet_d2_weighted_bce_best.pt --target-recall 0.95
```

The evaluator prints both policies and saves the complete report, checkpoint hash, and example error ECG IDs to:

```text
results/ecg_resnet_d2_weighted_bce_best_screening_recall90_test.json
```

## Current results

The following results are from held-out fold 10. All thresholds were selected on validation fold 9.

| Model and threshold policy | Macro AUROC | Macro AP | Macro F1 |
| --- | ---: | ---: | ---: |
| Original BCE, F1 thresholds | 0.9253 | 0.8131 | 0.7388 |
| Weighted BCE, F1 thresholds | **0.9271** | **0.8179** | **0.7441** |
| Original BCE, 90% recall screening | 0.9253 | 0.8131 | 0.6985 |
| Weighted BCE, 90% recall screening | **0.9271** | **0.8179** | **0.7003** |

Weighted BCE gives a small overall improvement. Its clearest gain is on the least common class, `HYP`: average precision increases from 0.659 to 0.676. It also improves the F1-threshold result from 0.590 to 0.599. `STTC` changes very little, so the weighted loss should be treated as a useful improvement rather than a complete solution to the imbalance.

With weighted BCE and F1 thresholds, the per-class test results are:

| Class | Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| NORM | 0.50 | 0.811 | 0.920 | 0.862 |
| MI | 0.65 | 0.763 | 0.738 | 0.750 |
| STTC | 0.70 | 0.742 | 0.777 | 0.759 |
| CD | 0.80 | 0.819 | 0.692 | 0.750 |
| HYP | 0.55 | 0.514 | 0.718 | 0.599 |

AUROC and average precision measure how well the model ranks examples across all thresholds. Precision, recall, and F1 depend on the selected threshold. This is why AUROC and average precision are identical for the two threshold policies on the same checkpoint.

## Project files

- `main.py` contains the dataset class, ResNet, training loop, weighted loss, early stopping, and validation reporting.
- `test.py` evaluates a checkpoint on validation and test folds and writes the final report.
- `metrics.py` contains metric calculation and threshold selection.
- `EDA.py` loads PTB-XL metadata and maps diagnostic statements to the five superclasses.
- `Models/` contains trained checkpoints and is ignored by Git.
- `results/` contains the smaller JSON experiment reports, which can be committed.
