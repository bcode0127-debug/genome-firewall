"""BUILD-TIME script (may read data/features). Enriches data/results/metrics.json
with full-test-set conformal coverage, reliability-curve bins, and novelty-gate
validation, so the Streamlit app can read everything from committed results files
without ever touching data/features/ or data/interim/.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"
VOCAB_MODEL_PATH = "data/features/vocab_model.json"
CONFORMAL_PATH = "models/conformal.json"
NOVELTY_PATH = "data/results/novelty.json"
METRICS_PATH = "data/results/metrics.json"

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
POSITIVE_CLASS = "Resistant"
NOMINAL = 0.90
JACCARD_THRESH = 0.1570
ABSENT_THRESH = 15
MIN_GROUP = 20
ST258 = "MLST.Klebsiella_pneumoniae.258"


def to_binary(s):
    return (s == POSITIVE_CLASS).astype(int)


def main():
    tt = pd.read_parquet(TRAIN_TABLE_PATH)
    tt["genome_id"] = tt["genome_id"].astype(str)
    splits = pd.read_csv(SPLITS_PATH, dtype={"genome_id": str})
    df = tt.merge(splits, on="genome_id", how="inner")
    with open(VOCAB_MODEL_PATH) as f:
        vocab_model = json.load(f)
    with open(CONFORMAL_PATH) as f:
        conformal = json.load(f)
    with open(NOVELTY_PATH) as f:
        novelty = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    nov = pd.DataFrame(novelty["per_genome"])
    nov["genome_id"] = nov["genome_id"].astype(str)
    nov["flagged"] = (nov["jaccard_nn_distance"] >= JACCARD_THRESH) | \
                     (nov["absent_feature_count"] >= ABSENT_THRESH)

    test_df = df[df["split"] == "test"].merge(nov[["genome_id", "flagged"]], on="genome_id")
    test_df["group"] = test_df["mlst"].where(test_df["mlst"].notna(),
                                             "UNTYPED_" + test_df["genome_id"])
    X_test = test_df[vocab_model].values.astype(float)

    metrics["positive_class"] = POSITIVE_CLASS
    metrics["nominal_coverage"] = NOMINAL

    for drug in DRUGS:
        model = joblib.load(f"models/{drug}.joblib")
        y = to_binary(test_df[drug]).values
        prob1 = model.predict_proba(X_test)[:, 1]
        pred = (prob1 >= 0.5).astype(int)

        alpha = conformal[drug]["alpha"]
        q_r = conformal[drug]["q_resistant"]
        q_s = conformal[drug]["q_susceptible"]
        inc_r = prob1 >= (1 - q_r)
        inc_s = (1 - prob1) >= (1 - q_s)
        set_size = inc_r.astype(int) + inc_s.astype(int)
        singleton = set_size == 1
        no_call_rate = float((~singleton).mean())
        true_incl = np.where(y == 1, inc_r, inc_s)

        # singleton correctness (non-abstained accuracy)
        singleton_label = np.where(inc_r & ~inc_s, 1, np.where(inc_s & ~inc_r, 0, -1))
        if singleton.sum():
            acc_non_abstained = float((singleton_label[singleton] == y[singleton]).mean())
        else:
            acc_non_abstained = None

        d = metrics["drugs"][drug]
        d["alpha"] = alpha
        d["conformal"] = {
            "q_resistant": q_r, "q_susceptible": q_s,
            "coverage_overall": float(true_incl.mean()),
            "coverage_resistant": float(true_incl[y == 1].mean()) if (y == 1).any() else None,
            "coverage_susceptible": float(true_incl[y == 0].mean()) if (y == 0).any() else None,
            "no_call_rate": no_call_rate,
            "nominal": NOMINAL,
        }
        # refresh test_overall's conformal-dependent fields to the final alpha
        d["test_overall"]["no_call_rate"] = no_call_rate
        d["test_overall"]["accuracy_on_non_abstained"] = acc_non_abstained

        # reliability curve (full test set)
        frac_pos, mean_pred = calibration_curve(y, prob1, n_bins=10, strategy="uniform")
        # bin counts
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(prob1, bins) - 1, 0, 9)
        counts = [int((idx == b).sum()) for b in range(10)]
        # map calibration_curve outputs (only non-empty bins) back with counts
        reliability = []
        nonempty = [b for b in range(10) if counts[b] > 0]
        for (mp, fp), b in zip(zip(mean_pred, frac_pos), nonempty):
            reliability.append({"mean_pred": float(mp), "obs_freq": float(fp),
                                "count": counts[b]})
        d["reliability"] = reliability

        # novelty validation on full test
        is_err = (pred != y)
        fl = test_df["flagged"].values
        n_fl = int(fl.sum())
        err_fl = float(is_err[fl].mean()) if n_fl else None
        err_nfl = float(is_err[~fl].mean()) if (~fl).any() else None
        total_err = int(is_err.sum())
        caught = int((is_err & fl).sum())
        d["novelty_validation"] = {
            "n_flagged": n_fl,
            "err_flagged": err_fl,
            "err_not_flagged": err_nfl,
            "total_errors": total_err,
            "errors_caught": caught,
            "discriminative": bool(err_fl is not None and err_nfl is not None and err_fl > err_nfl),
        }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Enriched {METRICS_PATH} with conformal eval, reliability bins, novelty validation.")
    for drug in DRUGS:
        c = metrics["drugs"][drug]["conformal"]
        nv = metrics["drugs"][drug]["novelty_validation"]
        print(f"  {drug}: alpha={metrics['drugs'][drug]['alpha']} "
              f"coverage={c['coverage_overall']:.4f} no_call={c['no_call_rate']:.4f} "
              f"novelty_discriminative={nv['discriminative']} "
              f"(err_fl={nv['err_flagged']}, err_nfl={nv['err_not_flagged']})")


if __name__ == "__main__":
    main()
