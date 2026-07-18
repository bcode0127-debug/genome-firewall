import json

import joblib
import numpy as np
import pandas as pd

TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"
VOCAB_MODEL_PATH = "data/features/vocab_model.json"
NOVELTY_PATH = "data/results/novelty.json"

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
POSITIVE_CLASS = "Resistant"
JACCARD_THRESH = 0.1570
ABSENT_THRESH = 15


def to_binary(series):
    return (series == POSITIVE_CLASS).astype(int)


def main():
    train_table = pd.read_parquet(TRAIN_TABLE_PATH)
    train_table["genome_id"] = train_table["genome_id"].astype(str)
    splits = pd.read_csv(SPLITS_PATH, dtype={"genome_id": str})
    df = train_table.merge(splits, on="genome_id", how="inner")
    with open(VOCAB_MODEL_PATH) as f:
        vocab_model = json.load(f)
    with open(NOVELTY_PATH) as f:
        novelty = json.load(f)

    nov_df = pd.DataFrame(novelty["per_genome"])
    nov_df["genome_id"] = nov_df["genome_id"].astype(str)
    nov_df["flagged"] = (nov_df["jaccard_nn_distance"] >= JACCARD_THRESH) | \
                        (nov_df["absent_feature_count"] >= ABSENT_THRESH)

    print(f"=== 2. Novelty validation (Jaccard >= {JACCARD_THRESH} OR "
          f"absent-count >= {ABSENT_THRESH}) ===")
    print(f"flagged genomes: {nov_df['flagged'].sum()} / {len(nov_df)}")
    print()

    test_df = df[df["split"] == "test"].merge(nov_df[["genome_id", "flagged"]], on="genome_id")

    for drug in DRUGS:
        model = joblib.load(f"models/{drug}.joblib")
        X_test = test_df[vocab_model].values.astype(float)
        y_test = to_binary(test_df[drug]).values
        prob1 = model.predict_proba(X_test)[:, 1]
        pred = (prob1 >= 0.5).astype(int)
        is_error = (pred != y_test)

        flagged_mask = test_df["flagged"].values
        n_flagged = int(flagged_mask.sum())
        n_not_flagged = int((~flagged_mask).sum())

        err_flagged = is_error[flagged_mask].mean() if n_flagged else float("nan")
        err_not_flagged = is_error[~flagged_mask].mean() if n_not_flagged else float("nan")

        total_errors = int(is_error.sum())
        errors_caught = int((is_error & flagged_mask).sum())
        catch_rate = errors_caught / total_errors if total_errors else float("nan")

        print(f"--- {drug} ---")
        print(f"  n_flagged={n_flagged}  n_not_flagged={n_not_flagged}")
        print(f"  error rate among FLAGGED:     {err_flagged:.4f}")
        print(f"  error rate among NOT-flagged: {err_not_flagged:.4f}")
        print(f"  total model errors on test: {total_errors}")
        print(f"  errors caught by flag: {errors_caught} / {total_errors} "
              f"({catch_rate:.4f} of all errors)")
        if pd.isna(err_flagged) or pd.isna(err_not_flagged) or err_flagged <= err_not_flagged:
            print("  VERDICT: flagged genomes do NOT show a clearly higher error rate "
                  "for this drug — the novelty gate is not discriminative here.")
        else:
            lift = err_flagged - err_not_flagged
            print(f"  VERDICT: flagged genomes DO show a higher error rate "
                  f"(+{lift:.4f} absolute) for this drug.")
        print()


if __name__ == "__main__":
    main()
