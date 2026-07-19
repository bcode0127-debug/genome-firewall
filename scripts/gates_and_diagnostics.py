import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import yaml

TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"
VOCAB_PATH = "data/features/vocab.json"
VOCAB_MODEL_PATH = "data/features/vocab_model.json"
MATRIX_PATH = "data/features/matrix.parquet"
CONFORMAL_PATH = "models/conformal.json"
DRUG_PROPS_PATH = "data/drug_properties.yaml"
SPGENE_DIR = "data/interim/spgene"
NOVELTY_OUT_PATH = "data/results/novelty.json"

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
POSITIVE_CLASS = "Resistant"
RESISTANCE_PROPERTY = "Antibiotic Resistance"
TARGET_PROPERTY = "Drug Target"
NOVELTY_TOP_PCT = 0.05


def to_binary(series):
    return (series == POSITIVE_CLASS).astype(int)


def load_data():
    train_table = pd.read_parquet(TRAIN_TABLE_PATH)
    train_table["genome_id"] = train_table["genome_id"].astype(str)
    splits = pd.read_csv(SPLITS_PATH, dtype={"genome_id": str})
    df = train_table.merge(splits, on="genome_id", how="inner")
    with open(VOCAB_MODEL_PATH) as f:
        vocab_model = json.load(f)
    with open(CONFORMAL_PATH) as f:
        conformal = json.load(f)
    return df, vocab_model, conformal


def get_test_probs(df, vocab_model, drug):
    test_df = df[df["split"] == "test"]
    X_test = test_df[vocab_model].values.astype(float)
    y_test = to_binary(test_df[drug]).values
    model = joblib.load(f"models/{drug}.joblib")
    prob1 = model.predict_proba(X_test)[:, 1]
    return test_df["genome_id"].values, y_test, prob1


def prediction_sets(prob1, q_resistant, q_susceptible):
    prob0 = 1 - prob1
    include_r = prob1 >= (1 - q_resistant)
    include_s = prob0 >= (1 - q_susceptible)
    return include_r, include_s


# =========================== PART A ===========================

def part_a(df, vocab_model, conformal):
    print("=" * 70)
    print("PART A — Conformal diagnostics")
    print("=" * 70)

    for drug in DRUGS:
        genome_ids, y_test, prob1 = get_test_probs(df, vocab_model, drug)
        q_r = conformal[drug]["q_resistant"]
        q_s = conformal[drug]["q_susceptible"]
        include_r, include_s = prediction_sets(prob1, q_r, q_s)

        true_included = np.where(y_test == 1, include_r, include_s)
        coverage_overall = true_included.mean()
        coverage_resistant = true_included[y_test == 1].mean() if (y_test == 1).any() else float("nan")
        coverage_susceptible = true_included[y_test == 0].mean() if (y_test == 0).any() else float("nan")

        print(f"\n--- {drug} ---")
        print("1. Empirical coverage (nominal = 0.90):")
        print(f"   overall:              {coverage_overall:.4f}  (n={len(y_test)})")
        print(f"   true=Resistant:       {coverage_resistant:.4f}  (n={(y_test == 1).sum()})")
        print(f"   true=Susceptible:     {coverage_susceptible:.4f}  (n={(y_test == 0).sum()})")

        set_size = include_r.astype(int) + include_s.astype(int)
        n_empty = int((set_size == 0).sum())
        n_both = int((set_size == 2).sum())
        n_singleton = int((set_size == 1).sum())
        print("2. No-call breakdown:")
        print(f"   EMPTY sets (neither class included): {n_empty} "
              f"({n_empty / len(y_test):.4f})")
        print(f"   BOTH-labels sets (both included):    {n_both} "
              f"({n_both / len(y_test):.4f})")
        print(f"   singleton (a call is made):          {n_singleton} "
              f"({n_singleton / len(y_test):.4f})")

        if drug == "gentamicin":
            print("3. Gentamicin alpha sensitivity:")
            cal_df = df[df["split"] == "calibration"]
            X_cal = cal_df[vocab_model].values.astype(float)
            y_cal = to_binary(cal_df[drug]).values
            model = joblib.load(f"models/{drug}.joblib")
            prob1_cal = model.predict_proba(X_cal)[:, 1]
            prob_true_cal = np.where(y_cal == 1, prob1_cal, 1 - prob1_cal)
            nonconformity_cal = 1 - prob_true_cal
            nc_r_cal = nonconformity_cal[y_cal == 1]
            nc_s_cal = nonconformity_cal[y_cal == 0]

            for alpha in [0.05, 0.20]:
                q_r_a = float(np.quantile(nc_r_cal, 1 - alpha))
                q_s_a = float(np.quantile(nc_s_cal, 1 - alpha))
                inc_r_a, inc_s_a = prediction_sets(prob1, q_r_a, q_s_a)
                size_a = inc_r_a.astype(int) + inc_s_a.astype(int)
                no_call_rate_a = (size_a != 1).mean()
                true_incl_a = np.where(y_test == 1, inc_r_a, inc_s_a)
                coverage_a = true_incl_a.mean()
                print(f"   alpha={alpha}: q_resistant={q_r_a:.4f} q_susceptible={q_s_a:.4f} "
                      f"no_call_rate={no_call_rate_a:.4f} coverage={coverage_a:.4f}")


# =========================== PART B ===========================

def compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def any_match(patterns, text):
    return any(p.search(text) for p in patterns)


def part_b4(drug_props):
    print("\n" + "=" * 70)
    print("PART B — Gates")
    print("=" * 70)
    print(f"\n4. Wrote {DRUG_PROPS_PATH} for: {list(drug_props.keys())}")
    for drug, props in drug_props.items():
        print(f"   {drug}: class={props['class']!r}, "
              f"{len(props['molecular_target_patterns'])} target patterns, "
              f"{len(props['resistance_determinant_patterns'])} determinant patterns")

    # sanity check: ceftazidime determinants must NOT match the narrow-spectrum
    # penicillinases blaSHV-11 / blaTEM-1 (not ESBLs), and must still match a
    # true ceftazidime-active ESBL allele.
    ceft_pats = compile_patterns(
        drug_props["ceftazidime"]["resistance_determinant_patterns"])
    print(f"   sanity check: any ceftazidime determinant matches 'SHV-11'? "
          f"{any_match(ceft_pats, 'shv-11')} (must be False)")
    print(f"   sanity check: any ceftazidime determinant matches 'TEM-1'? "
          f"{any_match(ceft_pats, 'tem-1')} (must be False)")
    print(f"   sanity check: any ceftazidime determinant matches 'SHV-12'? "
          f"{any_match(ceft_pats, 'shv-12')} (must be True)")
    print(f"   sanity check: any ceftazidime determinant matches 'CTX-M-15'? "
          f"{any_match(ceft_pats, 'ctx-m-15')} (must be True)")


def part_b5(df, drug_props, test_genome_ids):
    print("\n5. Target-presence gate (fires -> forced NO-CALL)")
    fires = {drug: [] for drug in DRUGS}

    for genome_id in test_genome_ids:
        path = os.path.join(SPGENE_DIR, f"{genome_id}.spgene.tab")
        try:
            gdf = pd.read_csv(path, sep="\t", low_memory=False,
                              usecols=["property", "product"])
        except (FileNotFoundError, ValueError):
            for drug in DRUGS:
                fires[drug].append(genome_id)  # no annotation at all -> can't confirm target
            continue
        targets = gdf.loc[gdf["property"] == TARGET_PROPERTY, "product"].astype(str)
        for drug in DRUGS:
            patterns = compile_patterns(drug_props[drug]["molecular_target_patterns"])
            present = targets.map(lambda t: any_match(patterns, t)).any()
            if not present:
                fires[drug].append(genome_id)

    for drug in DRUGS:
        n = len(fires[drug])
        print(f"   {drug}: fires on {n} / {len(test_genome_ids)} test genomes "
              f"({n / len(test_genome_ids):.4f}) — target pattern not found in "
              f"'{TARGET_PROPERTY}' annotations")
    return fires


def part_b6(df, vocab_model, vocab_full):
    print("\n6. Novelty gate diagnostics (not applied — report only)")

    matrix = pd.read_parquet(MATRIX_PATH)
    matrix["genome_id"] = matrix["genome_id"].astype(str)
    matrix = matrix.set_index("genome_id")

    vocab_model_set = set(vocab_model)
    pruned_cols = [c for c in vocab_full if c not in vocab_model_set]

    train_ids = df.loc[df["split"] == "train", "genome_id"].tolist()
    test_ids = df.loc[df["split"] == "test", "genome_id"].tolist()

    # (a) Jaccard nearest-neighbour distance in vocab_model feature space
    X_train = matrix.loc[train_ids, vocab_model].values.astype(np.float32)
    X_test = matrix.loc[test_ids, vocab_model].values.astype(np.float32)

    size_train = X_train.sum(axis=1)  # (n_train,)
    size_test = X_test.sum(axis=1)    # (n_test,)
    intersection = X_test @ X_train.T  # (n_test, n_train)
    union = size_test[:, None] + size_train[None, :] - intersection
    union_safe = np.where(union == 0, 1, union)
    jaccard_dist = 1 - (intersection / union_safe)
    jaccard_dist = np.where(union == 0, 0.0, jaccard_dist)
    nn_dist = jaccard_dist.min(axis=1)

    # (b) count of AMR features present in genome but absent from vocab_model
    X_test_pruned = matrix.loc[test_ids, pruned_cols].values.astype(np.float32)
    absent_count = X_test_pruned.sum(axis=1)

    def describe(arr, label):
        s = pd.Series(arr)
        print(f"   {label}:")
        print(f"     min={s.min():.4f} p25={s.quantile(.25):.4f} median={s.median():.4f} "
              f"p75={s.quantile(.75):.4f} p90={s.quantile(.90):.4f} p95={s.quantile(.95):.4f} "
              f"p99={s.quantile(.99):.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")

    describe(nn_dist, "Jaccard NN distance to nearest TRAIN genome")
    describe(absent_count, "count of AMR features absent from vocab_model.json")

    thresh_nn = float(np.quantile(nn_dist, 1 - NOVELTY_TOP_PCT))
    thresh_absent = float(np.quantile(absent_count, 1 - NOVELTY_TOP_PCT))
    n_flag_nn = int((nn_dist >= thresh_nn).sum())
    n_flag_absent = int((absent_count >= thresh_absent).sum())
    print(f"   top-5% threshold — Jaccard NN distance >= {thresh_nn:.4f} "
          f"flags {n_flag_nn} / {len(test_ids)} test genomes")
    print(f"   top-5% threshold — absent-feature count >= {thresh_absent:.4f} "
          f"flags {n_flag_absent} / {len(test_ids)} test genomes")
    print("   NOT applied to any prediction in this run.")

    out = {
        "top_pct": NOVELTY_TOP_PCT,
        "feature_space": "vocab_model.json (pruned, model-facing feature space)",
        "jaccard_nn_distance": {
            "threshold_top5pct": thresh_nn,
            "n_flagged": n_flag_nn,
        },
        "absent_feature_count": {
            "definition": "count of matrix.parquet columns (full 998-vocab) set to 1 "
                          "for this genome that are NOT in vocab_model.json's 422 survivors",
            "threshold_top5pct": thresh_absent,
            "n_flagged": n_flag_absent,
        },
        "per_genome": [
            {
                "genome_id": gid,
                "jaccard_nn_distance": float(nn_dist[i]),
                "absent_feature_count": int(absent_count[i]),
            }
            for i, gid in enumerate(test_ids)
        ],
    }
    with open(NOVELTY_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   Saved {NOVELTY_OUT_PATH}")


def main():
    df, vocab_model, conformal = load_data()
    with open(VOCAB_PATH) as f:
        vocab_full = json.load(f)
    with open(DRUG_PROPS_PATH) as f:
        drug_props = yaml.safe_load(f)

    part_a(df, vocab_model, conformal)

    part_b4(drug_props)
    test_genome_ids = df.loc[df["split"] == "test", "genome_id"].tolist()
    part_b5(df, drug_props, test_genome_ids)
    part_b6(df, vocab_model, vocab_full)


if __name__ == "__main__":
    main()
