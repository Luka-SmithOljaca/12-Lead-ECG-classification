"""Small PTB-XL metadata and signal inspection helpers."""

import ast
import os
from pathlib import Path

import pandas as pd
import wfdb


LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Data" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"


def get_data_dir(data_dir=None):
    path = Path(data_dir or os.environ.get("PTBXL_DATA_DIR", DEFAULT_DATA_DIR))
    if not (path / "ptbxl_database.csv").is_file():
        raise FileNotFoundError(f"PTB-XL metadata not found in {path}. Set PTBXL_DATA_DIR or pass --data-dir.")
    return path


def load_metadata(data_dir=None):
    data_dir = get_data_dir(data_dir)
    metadata = pd.read_csv(data_dir / "ptbxl_database.csv")
    statements = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1]

    def get_classes(codes_text):
        codes = ast.literal_eval(codes_text)
        return sorted({diagnostic.loc[code, "diagnostic_class"] for code in codes if code in diagnostic.index})

    metadata["classes"] = metadata["scp_codes"].apply(get_classes)
    for label in LABELS:
        metadata[label] = metadata["classes"].apply(lambda labels: label in labels)
    return metadata


if __name__ == "__main__":
    data_dir = get_data_dir()
    metadata = load_metadata(data_dir)
    signal, info = wfdb.rdsamp(str(data_dir / metadata.loc[0, "filename_lr"]))
    print("Metadata shape:", metadata.shape)
    print("Signal shape:", signal.shape)
    print("Sampling rate:", info["fs"])
    print("Recordings:", len(metadata))
    print("Patients:", metadata["patient_id"].nunique())
    print("Class counts:\n", metadata[LABELS].sum())
    print("Records with no diagnostic superclass:", (metadata[LABELS].sum(axis=1) == 0).sum())
    print("Records with multiple superclasses:", (metadata[LABELS].sum(axis=1) > 1).sum())
