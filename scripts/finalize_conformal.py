import json

import joblib
import numpy as np
import pandas as pd

TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"
VOCAB_MODEL_PATH = "data/features/vocab_model.json"
CONFORMAL_PATH = "models/conformal.json"

DRUGS_ALPHA = {"meropenem": 0.10, "ceftazidime": 0.10, "gentamicin": 0.05}
POSITIVE_CLASS = "Resistant"


def to_binary(series):
    return (series == POSITIVE_CLASS).astype(int)


def main():
    train_table = pd.read_parquet(TRAIN_TABLE_PATH)
    train_table["genome_id"] = train_table["genome_id"].astype(str)
    splits = pd.read_csv(SPLITS_PATH, dtype={"genome_id": str})
    df = train_table.merge(splits, on="genome_id", how="inner")
    with open(VOCAB_MODEL_PATH) as f:
        vocab_model = json.load(f)

    cal_df = df[df["split"] == "calibration"]
    test_df = df[df["split"] == "test"]
    X_cal = cal_df[vocab_model].values.astype(float)
    X_test = test_df[vocab_model].values.astype(float)

    conformal_out = {}
    print("=== 1. Per-drug alpha conformal refit ===")
    for drug, alpha in DRUGS_ALPHA.items():
        model = joblib.load(f"models/{drug}.joblib")
        y_cal = to_binary(cal_df[drug]).values
        y_test = to_binary(test_df[drug]).values

        prob1_cal = model.predict_proba(X_cal)[:, 1]
        prob_true_cal = np.where(y_cal == 1, prob1_cal, 1 - prob1_cal)
        nonconformity_cal = 1 - prob_true_cal
        nc_r = nonconformity_cal[y_cal == 1]
        nc_s = nonconformity_cal[y_cal == 0]
        q_r = float(np.quantile(nc_r, 1 - alpha))
        q_s = float(np.quantile(nc_s, 1 - alpha))

        conformal_out[drug] = {
            "alpha": alpha,
            "q_resistant": q_r,
            "n_resistant_calibration": int(len(nc_r)),
            "q_susceptible": q_s,
            "n_susceptible_calibration": int(len(nc_s)),
        }

        prob1_test = model.predict_proba(X_test)[:, 1]
        prob0_test = 1 - prob1_test
        include_r = prob1_test >= (1 - q_r)
        include_s = prob0_test >= (1 - q_s)
        set_size = include_r.astype(int) + include_s.astype(int)
        no_call_rate = float((set_size != 1).mean())

        true_included = np.where(y_test == 1, include_r, include_s)
        coverage = float(true_included.mean())
        coverage_r = float(true_included[y_test == 1].mean()) if (y_test == 1).any() else float("nan")
        coverage_s = float(true_included[y_test == 0].mean()) if (y_test == 0).any() else float("nan")

        print(f"\n--- {drug} (alpha={alpha}) ---")
        print(f"q_resistant:   {q_r:.4f}  (n_calibration={len(nc_r)})")
        print(f"q_susceptible: {q_s:.4f}  (n_calibration={len(nc_s)})")
        print(f"TEST no-call rate: {no_call_rate:.4f}")
        print(f"TEST empirical coverage: overall={coverage:.4f}  "
              f"true=Resistant={coverage_r:.4f}  true=Susceptible={coverage_s:.4f}")

    with open(CONFORMAL_PATH, "w") as f:
        json.dump(conformal_out, f, indent=2)
    print(f"\nSaved {CONFORMAL_PATH}")


if __name__ == "__main__":
    main()
