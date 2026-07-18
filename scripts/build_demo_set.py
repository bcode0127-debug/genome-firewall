import json
import random

import pandas as pd

TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"
VOCAB_PATH = "data/features/vocab.json"
VOCAB_MODEL_PATH = "data/features/vocab_model.json"
MATRIX_PATH = "data/features/matrix.parquet"
NOVELTY_PATH = "data/results/novelty.json"
OUT_PATH = "data/results/demo_genomes.parquet"

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
ST258 = "MLST.Klebsiella_pneumoniae.258"
JACCARD_THRESH = 0.1570
ABSENT_THRESH = 15
DEMO_SIZE = 40
MIN_ST258 = 6
MIN_FLAGGED = 4
SEED = 20240607


def main():
    train_table = pd.read_parquet(TRAIN_TABLE_PATH)
    train_table["genome_id"] = train_table["genome_id"].astype(str)
    splits = pd.read_csv(SPLITS_PATH, dtype={"genome_id": str})
    df = train_table.merge(splits, on="genome_id", how="inner")
    with open(VOCAB_MODEL_PATH) as f:
        vocab_model = json.load(f)
    with open(VOCAB_PATH) as f:
        vocab_full = json.load(f)
    with open(NOVELTY_PATH) as f:
        novelty = json.load(f)

    nov_df = pd.DataFrame(novelty["per_genome"])
    nov_df["genome_id"] = nov_df["genome_id"].astype(str)
    nov_df["flagged"] = (nov_df["jaccard_nn_distance"] >= JACCARD_THRESH) | \
                        (nov_df["absent_feature_count"] >= ABSENT_THRESH)

    test_df = df[df["split"] == "test"].merge(nov_df[["genome_id", "flagged"]], on="genome_id")
    test_df["group"] = test_df.apply(
        lambda r: r["mlst"] if isinstance(r["mlst"], str) and r["mlst"].strip() != ""
        else f"UNTYPED_{r['genome_id']}", axis=1,
    )

    rng = random.Random(SEED)

    st258_ids = test_df.loc[test_df["group"] == ST258, "genome_id"].tolist()
    flagged_ids = test_df.loc[test_df["flagged"], "genome_id"].tolist()

    print("=== 3. Demo set construction ===")
    print(f"ST258 candidates in test: {len(st258_ids)}")
    print(f"flagged-novel candidates in test: {len(flagged_ids)}")

    selected = set()

    # ST258: pick MIN_ST258, stratified by meropenem R/S where possible
    st258_r = [g for g in st258_ids
               if test_df.set_index("genome_id").loc[g, "meropenem"] == "Resistant"]
    st258_s = [g for g in st258_ids
               if test_df.set_index("genome_id").loc[g, "meropenem"] == "Susceptible"]
    rng.shuffle(st258_r)
    rng.shuffle(st258_s)
    half = MIN_ST258 // 2
    pick_st258 = st258_r[:half] + st258_s[:MIN_ST258 - half]
    if len(pick_st258) < MIN_ST258:  # backfill if one class is short
        remainder = [g for g in st258_ids if g not in pick_st258]
        rng.shuffle(remainder)
        pick_st258 += remainder[: MIN_ST258 - len(pick_st258)]
    selected.update(pick_st258)
    print(f"selected {len(pick_st258)} from ST258 "
          f"(meropenem R={sum(1 for g in pick_st258 if g in st258_r)}, "
          f"S={sum(1 for g in pick_st258 if g in st258_s)})")

    # Flagged-novel: pick MIN_FLAGGED not already selected (top up if overlap forces it)
    flagged_remaining = [g for g in flagged_ids if g not in selected]
    rng.shuffle(flagged_remaining)
    pick_flagged = flagged_remaining[:MIN_FLAGGED]
    already_flagged_in_st258 = [g for g in pick_st258 if g in flagged_ids]
    if len(pick_flagged) + len(already_flagged_in_st258) < MIN_FLAGGED:
        need = MIN_FLAGGED - len(pick_flagged) - len(already_flagged_in_st258)
        extra = [g for g in flagged_ids if g not in selected and g not in pick_flagged]
        pick_flagged += extra[:need]
    selected.update(pick_flagged)
    print(f"selected {len(pick_flagged)} additional flagged-novel genomes "
          f"(plus {len(already_flagged_in_st258)} already covered via ST258 overlap)")

    # Fill remainder toward DEMO_SIZE, balancing meropenem R/S overall
    lookup = test_df.set_index("genome_id")
    current_r = sum(1 for g in selected if lookup.loc[g, "meropenem"] == "Resistant")
    current_s = len(selected) - current_r
    target_r = DEMO_SIZE // 2

    # Prefer non-ST258 genomes for the fill step — ST258's >=6 minimum is already
    # met, and it's 65% of test, so unconstrained sampling would drown out
    # every other clone in a 40-genome demo. Only fall back to ST258 if the
    # non-ST258 pool can't cover the R/S balance.
    remaining_pool_nonst258 = [g for g in test_df["genome_id"]
                              if g not in selected and lookup.loc[g, "group"] != ST258]
    remaining_pool_st258 = [g for g in test_df["genome_id"]
                            if g not in selected and lookup.loc[g, "group"] == ST258]
    rng.shuffle(remaining_pool_nonst258)
    rng.shuffle(remaining_pool_st258)
    remaining_pool = remaining_pool_nonst258 + remaining_pool_st258  # non-ST258 first
    pool_r = [g for g in remaining_pool if lookup.loc[g, "meropenem"] == "Resistant"]
    pool_s = [g for g in remaining_pool if lookup.loc[g, "meropenem"] == "Susceptible"]

    need_total = DEMO_SIZE - len(selected)
    need_r = max(0, target_r - current_r)
    need_r = min(need_r, need_total, len(pool_r))
    fill = pool_r[:need_r]
    need_total -= len(fill)
    fill += pool_s[:need_total]
    if len(fill) < need_total + len(fill):  # backfill if a bucket ran short
        leftover_needed = DEMO_SIZE - len(selected) - len(fill)
        if leftover_needed > 0:
            backup = [g for g in remaining_pool if g not in fill]
            fill += backup[:leftover_needed]

    selected.update(fill)
    selected = list(selected)[:DEMO_SIZE]
    if len(selected) < DEMO_SIZE:
        backup = [g for g in test_df["genome_id"] if g not in selected]
        rng.shuffle(backup)
        selected += backup[: DEMO_SIZE - len(selected)]

    demo_ids = selected
    demo_df = test_df[test_df["genome_id"].isin(demo_ids)].copy()

    n_r = (demo_df["meropenem"] == "Resistant").sum()
    n_s = (demo_df["meropenem"] == "Susceptible").sum()
    n_st258 = (demo_df["group"] == ST258).sum()
    n_flagged = demo_df["flagged"].sum()
    print(f"\nfinal demo set: {len(demo_df)} genomes")
    print(f"  meropenem: Resistant={n_r} Susceptible={n_s}")
    print(f"  from ST258: {n_st258} (>= {MIN_ST258} required)")
    print(f"  flagged novel: {n_flagged} (>= {MIN_FLAGGED} required)")

    # AR gene list per genome from the full 998-vocab matrix (raw AR gene identity)
    matrix = pd.read_parquet(MATRIX_PATH)
    matrix["genome_id"] = matrix["genome_id"].astype(str)
    matrix = matrix.set_index("genome_id")

    def ar_genes(genome_id):
        row = matrix.loc[genome_id, vocab_full]
        return sorted(row[row == 1].index.tolist())

    out_rows = []
    for _, r in demo_df.iterrows():
        gid = r["genome_id"]
        row = {
            "genome_id": gid,
            "mlst": r["mlst"] if isinstance(r["mlst"], str) else None,
            "meropenem": r["meropenem"],
            "ceftazidime": r["ceftazidime"],
            "gentamicin": r["gentamicin"],
            "flagged_novel": bool(r["flagged"]),
            "ar_genes": ar_genes(gid),
        }
        for col in vocab_model:  # exact order, 422 columns
            row[col] = int(r[col])
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    ordered_cols = ["genome_id", "mlst", "meropenem", "ceftazidime", "gentamicin",
                    "flagged_novel", "ar_genes"] + vocab_model
    out_df = out_df[ordered_cols]
    out_df.to_parquet(OUT_PATH, index=False)

    import os
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nSaved {OUT_PATH} — {len(out_df)} rows x {len(ordered_cols)} cols, "
          f"{size_kb:.1f} KB")


if __name__ == "__main__":
    main()
