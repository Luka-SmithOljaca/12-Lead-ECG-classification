"""Train and evaluate several seeds per loss, then report mean ± std on the test fold.

A single run cannot show whether a small difference between losses is real or
just seed-to-seed noise. This script repeats training for each seed and loss,
evaluates every checkpoint with validation-only thresholds, and compares the
losses seed by seed.

Finished runs are reused, so an interrupted sweep can be resumed. Use --rerun
to train everything again.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data import LABELS
from engine import get_device
from evaluate import evaluate_checkpoint, save_record
from main import LOSSES, PROJECT_DIR, add_training_arguments, config_from_args, load_datasets, run_name, train_model

SUMMARY_METRICS = {
    "macro_auroc": ("test", "macro_auroc"),
    "macro_ap": ("test", "macro_average_precision"),
    "macro_f1_f1_thresholds": ("f1_threshold_baseline_on_same_checkpoint", "macro_f1"),
    "macro_f1_screening": ("test", "macro_f1"),
}


def extract_metrics(record):
    values = {name: record[section][key] for name, (section, key) in SUMMARY_METRICS.items()}
    for label in LABELS:
        values[f"{label}_ap"] = record["test"]["per_class"][label]["average_precision"]
    return values


def mean_std(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {"mean": float(values.mean()), "std": std, "n": len(values)}


def summarise(results, seeds, losses):
    """results[loss][seed] -> metric dict. Returns per-loss and paired summaries."""
    metric_names = list(next(iter(results[losses[0]].values())).keys())
    summary = {
        loss: {m: mean_std([results[loss][s][m] for s in seeds]) for m in metric_names}
        for loss in losses
    }
    if set(LOSSES) <= set(losses):
        # Same seed -> same initialisation and batch order, so pair the runs.
        summary["weighted_minus_unweighted"] = {
            m: mean_std([results["weighted_bce"][s][m] - results["bce"][s][m] for s in seeds])
            for m in metric_names
        }
    return summary


def print_summary(summary, losses):
    rows = list(SUMMARY_METRICS) + ["HYP_ap"]
    print("\nTest fold 10, mean ± std over seeds")
    print(f"{'Metric':26}" + "".join(f"{loss:>22}" for loss in losses))
    for metric in rows:
        cells = "".join(
            f"{summary[loss][metric]['mean']:>13.4f} ± {summary[loss][metric]['std']:.4f}" for loss in losses
        )
        print(f"{metric:26}{cells}")
    if "weighted_minus_unweighted" in summary:
        print("\nPaired difference (weighted - unweighted, same seed)")
        for metric in rows:
            d = summary["weighted_minus_unweighted"][metric]
            print(f"{metric:26}{d['mean']:>+13.4f} ± {d['std']:.4f}")


def main():
    parser = add_training_arguments(argparse.ArgumentParser(description=__doc__,
                                                            formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--losses", choices=LOSSES, nargs="+", default=list(LOSSES))
    parser.add_argument("--rerun", action="store_true", help="Retrain runs that already have results")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}. Loading PTB-XL...")
    train_dataset, val_dataset, test_dataset = load_datasets(args.data_dir)

    results = {loss: {} for loss in args.losses}
    for loss in args.losses:
        args.loss = loss
        for seed in args.seeds:
            config = config_from_args(args, seed)
            name = run_name(config)
            checkpoint = PROJECT_DIR / "Models" / f"{name}.pt"
            result_path = PROJECT_DIR / "results" / f"{name}_screening_recall{round(args.target_recall * 100)}_test.json"

            if result_path.is_file() and checkpoint.is_file() and not args.rerun:
                print(f"{name}: reusing saved results")
                record = json.loads(result_path.read_text())
            else:
                print(f"{name}: training")
                checkpoint, _ = train_model(config, train_dataset, val_dataset, device, verbose=False)
                record = evaluate_checkpoint(checkpoint, val_dataset, test_dataset, device, args.target_recall)
                save_record(record, checkpoint, args.target_recall)

            results[loss][seed] = extract_metrics(record)
            print(f"  test macro AUROC = {results[loss][seed]['macro_auroc']:.4f}")

    summary = summarise(results, args.seeds, args.losses)
    print_summary(summary, args.losses)
    output = PROJECT_DIR / "results" / "seed_summary.json"
    output.write_text(json.dumps({"seeds": args.seeds, "per_run": results, "summary": summary}, indent=2))
    print(f"\nSaved: {output.relative_to(PROJECT_DIR).as_posix()}")


if __name__ == "__main__":
    main()
