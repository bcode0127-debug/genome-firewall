import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"
VOCAB_MODEL_PATH = "data/features/vocab_model.json"
CONFORMAL_PATH = "models/conformal.json"
METRICS_PATH = "data/results/metrics.json"

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
POSITIVE_CLASS = "Resistant"  # positive class throughout: Resistant = 1
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_FOLDS = 5
ALPHA = 0.10
MIN_GROUP_SIZE_FOR_TEST_REPORT = 20
TUNING_SCORER = "neg_log_loss (mean log_loss across GroupKFold(5) folds, lower is better)"


def is_blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def to_binary(series):
    return (series == POSITIVE_CLASS).astype(int)


def log_loss_safe(y_true, prob1, eps=1e-15):
    p = np.clip(prob1, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def tune_c(X, y, groups, class_weight):
    gkf = GroupKFold(n_splits=N_FOLDS)
    best_c, best_loss = None, np.inf
    per_c_losses = {}
    for c in C_GRID:
        fold_losses = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            y_tr, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            clf = LogisticRegression(penalty="l2", C=c, max_iter=2000,
                                     class_weight=class_weight)
            clf.fit(X[train_idx], y_tr)
            prob1 = clf.predict_proba(X[val_idx])[:, 1]
            fold_losses.append(log_loss_safe(y_val, prob1))
        mean_loss = np.mean(fold_losses) if fold_losses else np.inf
        per_c_losses[c] = mean_loss
        if mean_loss < best_loss:
            best_loss, best_c = mean_loss, c
    return best_c, best_loss, per_c_losses


def eval_probs(y_true, prob1, threshold=0.5):
    pred = (prob1 >= threshold).astype(int)
    brier = brier_score_loss(y_true, prob1)
    try:
        auroc = roc_auc_score(y_true, prob1)
    except ValueError:
        auroc = float("nan")
    try:
        prauc = average_precision_score(y_true, prob1)
    except ValueError:
        prauc = float("nan")
    recall_resistant = recall_score(y_true, pred, pos_label=1, zero_division=0)
    recall_susceptible = recall_score(y_true, pred, pos_label=0, zero_division=0)
    return {
        "brier": brier, "auroc": auroc, "pr_auc": prauc,
        "recall_resistant": recall_resistant, "recall_susceptible": recall_susceptible,
    }


def main():
    # === 1. Load & join ===
    train_table = pd.read_parquet(TRAIN_TABLE_PATH)
    train_table["genome_id"] = train_table["genome_id"].astype(str)
    splits = pd.read_csv(SPLITS_PATH, dtype={"genome_id": str})
    df = train_table.merge(splits, on="genome_id", how="inner")
    print(f"joined rows: {len(df)} (train_table={len(train_table)}, splits={len(splits)})")
    print(df["split"].value_counts().to_string())
    print()

    df["group"] = df.apply(
        lambda r: r["mlst"] if not is_blank(r["mlst"]) else f"UNTYPED_{r['genome_id']}",
        axis=1,
    )

    all_feature_cols = [c for c in train_table.columns
                        if c not in ("genome_id",) + tuple(DRUGS)]
    print(f"feature columns before pruning: {len(all_feature_cols)}")

    # === 2. Feature pruning on TRAIN ROWS ONLY ===
    train_mask = df["split"] == "train"
    train_df = df[train_mask]
    presence_rate = train_df[all_feature_cols].mean(axis=0)  # binary 0/1 cols -> mean = presence rate

    drop_high = presence_rate[presence_rate > 0.95].index.tolist()
    drop_low = presence_rate[presence_rate < 0.01].index.tolist()
    survivors = [c for c in all_feature_cols if c not in drop_high and c not in drop_low]

    print("=== 2. Feature pruning (computed on train rows only) ===")
    print(f"dropped (present in >95% of train genomes): {len(drop_high)}")
    print(f"dropped (present in <1% of train genomes): {len(drop_low)}")
    print(f"survivors: {len(survivors)} "
          f"(before={len(all_feature_cols)}, after={len(survivors)})")
    with open(VOCAB_MODEL_PATH, "w") as f:
        json.dump(sorted(survivors), f, indent=2)
    print(f"Saved {VOCAB_MODEL_PATH}")
    print()

    X_all = df[survivors].values.astype(float)
    splits_idx = {s: df.index[df["split"] == s].to_numpy() for s in ["train", "calibration", "test"]}
    pos = {idx: i for i, idx in enumerate(df.index)}

    def subset(split_name):
        rows = splits_idx[split_name]
        pos_i = [pos[r] for r in rows]
        return X_all[pos_i], rows

    X_train, train_rows = subset("train")
    X_cal, cal_rows = subset("calibration")
    X_test, test_rows = subset("test")
    groups_train = df.loc[train_rows, "group"].values

    conformal_out = {}
    metrics_out = {"drugs": {}}

    for drug in DRUGS:
        print(f"##################### DRUG: {drug} #####################")
        y_train = to_binary(df.loc[train_rows, drug]).values
        y_cal = to_binary(df.loc[cal_rows, drug]).values
        y_test = to_binary(df.loc[test_rows, drug]).values

        # === 3. Tune C for both variants via GroupKFold(5) on mlst ===
        print(f"tuning scorer: {TUNING_SCORER}")
        variants = {}
        for name, cw in [("A_unweighted", None), ("B_balanced", "balanced")]:
            best_c, best_loss, per_c = tune_c(X_train, y_train, groups_train, cw)
            clf = LogisticRegression(penalty="l2", C=best_c, max_iter=2000, class_weight=cw)
            clf.fit(X_train, y_train)
            prob1_cal = clf.predict_proba(X_cal)[:, 1]
            ev = eval_probs(y_cal, prob1_cal)
            variants[name] = {"C": best_c, "cv_loss": best_loss, "model": clf, **ev}
            print(f"  {name}: best_C={best_c} cv_loss={best_loss:.4f}")

        print()
        print(f"--- {drug}: variant comparison on CALIBRATION split ---")
        header = f"{'variant':<14}{'C':>8}{'brier':>10}{'auroc':>10}{'pr_auc':>10}{'recall_R':>10}{'recall_S':>10}"
        print(header)
        for name, v in variants.items():
            print(f"{name:<14}{v['C']:>8}{v['brier']:>10.4f}{v['auroc']:>10.4f}"
                  f"{v['pr_auc']:>10.4f}{v['recall_resistant']:>10.4f}{v['recall_susceptible']:>10.4f}")

        names = list(variants.keys())
        lower_brier_name = min(names, key=lambda n: variants[n]["brier"])
        other_name = [n for n in names if n != lower_brier_name][0]
        recall_gap = variants[other_name]["recall_resistant"] - variants[lower_brier_name]["recall_resistant"]
        if recall_gap > 0.10:
            winner_name = other_name
            reason = (f"{lower_brier_name} has the better Brier "
                      f"({variants[lower_brier_name]['brier']:.4f} vs {variants[other_name]['brier']:.4f}), "
                      f"but its resistant-recall ({variants[lower_brier_name]['recall_resistant']:.4f}) is "
                      f"{recall_gap:.4f} worse than {other_name}'s ({variants[other_name]['recall_resistant']:.4f}) "
                      f"— over the 0.10 threshold, so {other_name} wins instead.")
        else:
            winner_name = lower_brier_name
            reason = (f"{lower_brier_name} has the better Brier "
                      f"({variants[lower_brier_name]['brier']:.4f} vs {variants[other_name]['brier']:.4f}); "
                      f"resistant-recall gap vs {other_name} is {abs(recall_gap):.4f}, "
                      f"within the 0.10 tolerance, so Brier decides.")
        print(f"WINNER: {winner_name} — {reason}")
        print()

        winning_model = variants[winner_name]["model"]
        brier_before = variants[winner_name]["brier"]

        # === 4. Isotonic calibration on CALIBRATION split ===
        # sklearn >=1.6 removed cv='prefit'; FrozenEstimator is the replacement
        # for calibrating an already-fitted model without refitting it.
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(winning_model), method="isotonic")
        calibrated.fit(X_cal, y_cal)
        prob1_cal_after = calibrated.predict_proba(X_cal)[:, 1]
        brier_after = brier_score_loss(y_cal, prob1_cal_after)
        print(f"--- {drug}: isotonic calibration ---")
        print(f"Brier on calibration split BEFORE calibration: {brier_before:.4f}")
        print(f"Brier on calibration split AFTER  calibration: {brier_after:.4f} "
              f"(in-sample check — same split the calibrator was fit on)")
        model_path = f"models/{drug}.joblib"
        joblib.dump(calibrated, model_path)
        print(f"Saved {model_path}")
        print()

        # === 5. Class-conditional conformal on calibration split ===
        prob1_cal_calibrated = calibrated.predict_proba(X_cal)[:, 1]
        prob_true_class = np.where(y_cal == 1, prob1_cal_calibrated, 1 - prob1_cal_calibrated)
        nonconformity = 1 - prob_true_class

        nc_resistant = nonconformity[y_cal == 1]
        nc_susceptible = nonconformity[y_cal == 0]
        q_resistant = float(np.quantile(nc_resistant, 1 - ALPHA)) if len(nc_resistant) else None
        q_susceptible = float(np.quantile(nc_susceptible, 1 - ALPHA)) if len(nc_susceptible) else None

        print(f"--- {drug}: class-conditional conformal (alpha={ALPHA}) ---")
        print(f"q_resistant (0.90 quantile of nonconformity | true=Resistant): "
              f"{q_resistant:.4f}  (n={len(nc_resistant)})")
        print(f"q_susceptible (0.90 quantile of nonconformity | true=Susceptible): "
              f"{q_susceptible:.4f}  (n={len(nc_susceptible)})")
        print()

        conformal_out[drug] = {
            "alpha": ALPHA,
            "q_resistant": q_resistant,
            "n_resistant_calibration": int(len(nc_resistant)),
            "q_susceptible": q_susceptible,
            "n_susceptible_calibration": int(len(nc_susceptible)),
        }

        # === 6. Metrics on TEST ===
        prob1_test = calibrated.predict_proba(X_test)[:, 1]
        prob0_test = 1 - prob1_test

        include_resistant = prob1_test >= (1 - q_resistant)
        include_susceptible = prob0_test >= (1 - q_susceptible)
        set_size = include_resistant.astype(int) + include_susceptible.astype(int)
        is_singleton = set_size == 1
        singleton_label = np.where(include_resistant & ~include_susceptible, 1,
                                    np.where(include_susceptible & ~include_resistant, 0, -1))

        pred_point = (prob1_test >= 0.5).astype(int)

        def metrics_for_subset(y_sub, prob1_sub, pred_sub, singleton_mask_sub, singleton_label_sub, label):
            n = len(y_sub)
            out = {"n": int(n)}
            if n == 0:
                return out
            out["balanced_accuracy"] = float(balanced_accuracy_score(y_sub, pred_sub))
            out["recall_resistant"] = float(recall_score(y_sub, pred_sub, pos_label=1, zero_division=0))
            out["recall_susceptible"] = float(recall_score(y_sub, pred_sub, pos_label=0, zero_division=0))
            out["f1_resistant_positive"] = float(f1_score(y_sub, pred_sub, pos_label=1, zero_division=0))
            try:
                out["auroc"] = float(roc_auc_score(y_sub, prob1_sub))
            except ValueError:
                out["auroc"] = None
            try:
                out["pr_auc_positive_class_resistant"] = float(average_precision_score(y_sub, prob1_sub))
            except ValueError:
                out["pr_auc_positive_class_resistant"] = None
            out["brier"] = float(brier_score_loss(y_sub, prob1_sub))
            out["no_call_rate"] = float((~singleton_mask_sub).mean())
            if singleton_mask_sub.sum() > 0:
                correct = (singleton_label_sub[singleton_mask_sub] == y_sub[singleton_mask_sub])
                out["accuracy_on_non_abstained"] = float(correct.mean())
            else:
                out["accuracy_on_non_abstained"] = None
            return out

        overall = metrics_for_subset(y_test, prob1_test, pred_point, is_singleton, singleton_label, "TEST overall")
        print(f"--- {drug}: TEST metrics (overall, n={overall['n']}, positive class=Resistant) ---")
        for k, v in overall.items():
            print(f"  {k}: {v}")
        print()

        # Per-MLST-group (test), >=20 members, ST258 first
        test_group_series = df.loc[test_rows, "group"]
        group_sizes = test_group_series.value_counts()
        big_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE_FOR_TEST_REPORT].index.tolist()
        if "MLST.Klebsiella_pneumoniae.258" in big_groups:
            big_groups.remove("MLST.Klebsiella_pneumoniae.258")
            ordered_groups = ["MLST.Klebsiella_pneumoniae.258"] + sorted(
                big_groups, key=lambda g: -group_sizes[g])
        else:
            ordered_groups = sorted(big_groups, key=lambda g: -group_sizes[g])

        per_group_out = {}
        print(f"--- {drug}: TEST metrics per MLST group (>= {MIN_GROUP_SIZE_FOR_TEST_REPORT} members, "
              f"ST258 first) ---")
        test_group_arr = test_group_series.values
        for g in ordered_groups:
            mask = test_group_arr == g
            gm = metrics_for_subset(y_test[mask], prob1_test[mask], pred_point[mask],
                                    is_singleton[mask], singleton_label[mask], g)
            per_group_out[g] = gm
            print(f"  [{g}] n={gm['n']}")
            for k, v in gm.items():
                if k == "n":
                    continue
                print(f"    {k}: {v}")
        print()

        metrics_out["drugs"][drug] = {
            "winner_variant": winner_name,
            "winner_C": variants[winner_name]["C"],
            "brier_calibration_before": brier_before,
            "brier_calibration_after": brier_after,
            "test_overall": overall,
            "test_by_group": per_group_out,
        }

    with open(CONFORMAL_PATH, "w") as f:
        json.dump(conformal_out, f, indent=2)
    print(f"Saved {CONFORMAL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
