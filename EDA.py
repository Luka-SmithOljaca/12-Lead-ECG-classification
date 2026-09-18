from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wfdb

data_dir = Path(r"C:\Users\lukas\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

metadata = pd.read_csv(data_dir / "ptbxl_database.csv")


record_path = data_dir / metadata.loc[0, "filename_lr"]
signal, info = wfdb.rdsamp(str(record_path))


headers = metadata.columns.tolist()

import ast

statements = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
diagnostic = statements[statements["diagnostic"] == 1]

def get_classes(codes_text):
    codes = ast.literal_eval(codes_text)
    classes = set()

    for code in codes:
        if code in diagnostic.index:
            classes.add(diagnostic.loc[code, "diagnostic_class"])

    return sorted(classes)

metadata["classes"] = metadata["scp_codes"].apply(get_classes)



classes = ["NORM", "MI", "STTC", "CD", "HYP"]

for label in classes:
    metadata[label] = metadata["classes"].apply(lambda labels: label in labels)

if __name__ == "__main__":
    print(metadata.shape)
    print(metadata[["ecg_id", "filename_lr", "strat_fold"]].head())

    print(signal.shape)
    print(info["fs"], info["sig_name"])

    print(headers)
    print(metadata[["ecg_id", "patient_id", "scp_codes", "strat_fold"]].head())
    print("Recordings:", len(metadata))
    print("Patients:", metadata["patient_id"].nunique())

    print(metadata[["ecg_id", "scp_codes", "classes"]].head(10))

    print(metadata[classes].sum().sort_values(ascending=False))
    print("Records with no diagnostic superclass:",
          (metadata[classes].sum(axis=1) == 0).sum())
    print("Records with multiple superclasses:",
          (metadata[classes].sum(axis=1) > 1).sum())
