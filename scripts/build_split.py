import random

import pandas as pd

META_PATH = "data/raw/genome_metadata.txt"
TRAIN_TABLE_PATH = "data/features/train_table.parquet"
SPLITS_PATH = "data/features/splits.csv"

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
BOUNDS = (0.15, 0.85)  # applies to ALL THREE drugs, in every split
TARGET_FRACS = {"test": 0.20, "calibration": 0.20, "train": 0.60}
FORCE_ST258 = "MLST.Klebsiella_pneumoniae.258"
SEED = 12345
MAX_TRIALS = 20000

MID_MIN, MID_MAX = 20, 150
MIN_MID_GROUPS_IN_TEST = 5
MIN_SINGLETONS_IN_TEST = 20
CAL_MAX_GROUP_SHARE = 0.15


def is_blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def resistant_fraction(genome_ids_by_group, groups, drug):
    size = sum(groups[g]["size"] for g in genome_ids_by_group)
    if size == 0:
        return None, 0
    resistant = sum(groups[g]["resistant"][drug] for g in genome_ids_by_group)
    return resistant / size, size


def main():
    train_table = pd.read_parquet(TRAIN_TABLE_PATH, columns=["genome_id"] + DRUGS)
    print(f"train_table.parquet rows: {len(train_table)}")

    meta = pd.read_csv(META_PATH, sep="\t", low_memory=False,
                        dtype={"genome_id": str}, usecols=["genome_id", "mlst"])
    train_table["genome_id"] = train_table["genome_id"].astype(str)
    target_ids = set(train_table["genome_id"])
    meta_sub = meta[meta["genome_id"].isin(target_ids)][["genome_id", "mlst"]]

    joined = train_table.merge(meta_sub, on="genome_id", how="left")
    n_typed = (~joined["mlst"].map(is_blank)).sum()
    n_untyped = joined["mlst"].map(is_blank).sum()
    print("=== Join coverage (same as before) ===")
    print(f"typed: {n_typed} ({n_typed / len(joined):.4f})  "
          f"untyped: {n_untyped} ({n_untyped / len(joined):.4f}) -> singleton groups")
    print()

    joined["group"] = joined.apply(
        lambda r: r["mlst"] if not is_blank(r["mlst"]) else f"UNTYPED_{r['genome_id']}",
        axis=1,
    )

    groups = {}
    for group_id, sub in joined.groupby("group"):
        groups[group_id] = {
            "genome_ids": sub["genome_id"].tolist(),
            "size": len(sub),
            "resistant": {d: int((sub[d] == "Resistant").sum()) for d in DRUGS},
        }
    all_group_ids = list(groups.keys())
    n_total = len(joined)
    print(f"total groups: {len(all_group_ids)}")

    if FORCE_ST258 not in groups:
        raise SystemExit("ST258 not present in this cohort — cannot force per instruction")

    st258_frac, st258_size = resistant_fraction([FORCE_ST258], groups, "meropenem")
    print(f"=== 1. ST258 forced into test ===")
    print(f"ST258 size: {st258_size} genomes")
    print(f"ST258-only meropenem resistant fraction: {st258_frac:.4f}")
    print()

    mid_groups = [g for g in all_group_ids
                  if g != FORCE_ST258 and MID_MIN <= groups[g]["size"] <= MID_MAX]
    singleton_groups = [g for g in all_group_ids if groups[g]["size"] == 1]
    remaining_pool_base = [g for g in all_group_ids
                           if g != FORCE_ST258]

    print(f"mid-size groups available (20-150 members): {len(mid_groups)}")
    print(f"singleton/untyped groups available: {len(singleton_groups)}")
    print()

    def bounds_ok(frac):
        return frac is not None and BOUNDS[0] <= frac <= BOUNDS[1]

    def build_trial(trial):
        rng = random.Random(SEED + trial)

        mid_pool = mid_groups[:]
        rng.shuffle(mid_pool)
        n_mid = rng.randint(MIN_MID_GROUPS_IN_TEST, MIN_MID_GROUPS_IN_TEST + 4)
        chosen_mid = mid_pool[:n_mid]

        singleton_pool = [g for g in singleton_groups if g not in chosen_mid]
        rng.shuffle(singleton_pool)
        n_single = rng.randint(MIN_SINGLETONS_IN_TEST, MIN_SINGLETONS_IN_TEST + 15)
        chosen_single = singleton_pool[:n_single]

        test_groups = [FORCE_ST258] + chosen_mid + chosen_single
        test_size = sum(groups[g]["size"] for g in test_groups)

        # top up / trim toward ~20% using small random groups from the leftover pool
        target_test_size = TARGET_FRACS["test"] * n_total
        leftover = [g for g in remaining_pool_base if g not in test_groups]
        rng.shuffle(leftover)
        for g in leftover:
            if test_size >= target_test_size - 20:
                break
            if groups[g]["size"] <= 60:  # keep additions modest so mix stays diverse
                test_groups.append(g)
                test_size += groups[g]["size"]

        # Check test composition + bounds
        n_mid_in_test = sum(1 for g in test_groups if g in mid_groups)
        n_single_in_test = sum(1 for g in test_groups if g in singleton_groups)
        if n_mid_in_test < MIN_MID_GROUPS_IN_TEST or n_single_in_test < MIN_SINGLETONS_IN_TEST:
            return None
        test_size_frac = test_size / n_total
        if not (0.15 <= test_size_frac <= 0.25):
            return None
        for d in DRUGS:
            frac, _ = resistant_fraction(test_groups, groups, d)
            if not bounds_ok(frac):
                return None

        # Assign remaining groups to calibration / train.
        remaining = [g for g in remaining_pool_base if g not in test_groups]
        rng.shuffle(remaining)
        cal_target_size = TARGET_FRACS["calibration"] * n_total
        cal_cap = CAL_MAX_GROUP_SHARE * cal_target_size

        cal_groups, train_groups = [], []
        cal_size, train_size = 0, 0
        for g in remaining:
            size = groups[g]["size"]
            cal_deficit = TARGET_FRACS["calibration"] - (cal_size / n_total)
            train_deficit = TARGET_FRACS["train"] - (train_size / n_total)
            can_go_cal = size <= cal_cap
            if can_go_cal and cal_deficit >= train_deficit:
                cal_groups.append(g)
                cal_size += size
            else:
                train_groups.append(g)
                train_size += size

        for d in DRUGS:
            for split_groups in (cal_groups, train_groups):
                frac, _ = resistant_fraction(split_groups, groups, d)
                if not bounds_ok(frac):
                    return None

        return {
            "test": test_groups,
            "calibration": cal_groups,
            "train": train_groups,
        }

    result = None
    trial_used = None
    for trial in range(MAX_TRIALS):
        result = build_trial(trial)
        if result is not None:
            trial_used = trial
            break

    print("=== 2/3/4. Search result ===")
    if result is None:
        raise SystemExit(f"No satisfying assignment found in {MAX_TRIALS} trials.")
    print(f"solved on trial {trial_used} (seed={SEED})")
    print()

    assignment = {}
    for split, gs in result.items():
        for g in gs:
            assignment[g] = split
    joined["split"] = joined["group"].map(assignment)

    print("=== Per-split summary ===")
    for split in ["train", "calibration", "test"]:
        sub = joined[joined["split"] == split]
        n_groups = sub["group"].nunique()
        group_sizes = sub["group"].value_counts()
        largest_group = group_sizes.index[0]
        largest_share = group_sizes.iloc[0] / len(sub)

        print(f"--- {split} ---")
        print(f"genome count: {len(sub)} ({len(sub) / n_total:.4f} of total)")
        print(f"group count: {n_groups}")
        for d in DRUGS:
            frac = (sub[d] == "Resistant").mean()
            ok = "OK" if BOUNDS[0] <= frac <= BOUNDS[1] else "OUT OF BOUNDS"
            print(f"  {d} resistant fraction: {frac:.4f} {ok}")
        print(f"largest group: {largest_group} — {group_sizes.iloc[0]} genomes "
              f"({largest_share:.4f} of split)")
        if split == "calibration" and largest_share > CAL_MAX_GROUP_SHARE:
            print(f"  FLAG: largest group exceeds {CAL_MAX_GROUP_SHARE:.0%} of calibration")
        top10 = group_sizes.head(10)
        print("  10 largest groups:")
        for g, cnt in top10.items():
            print(f"    {g}: {cnt}")
        print()

    print("=== Per-group meropenem resistant-fraction heterogeneity ===")
    for split in ["calibration", "test"]:
        sub = joined[joined["split"] == split]
        per_group = sub.groupby("group").apply(
            lambda s: (s["meropenem"] == "Resistant").mean()
        )
        print(f"{split}: mean={per_group.mean():.4f}  std={per_group.std():.4f}  "
              f"(n_groups={len(per_group)})")
    print()

    if joined["split"].isna().any():
        n_missing = joined["split"].isna().sum()
        print(f"WARNING: {n_missing} genomes have no split assignment")

    out = joined[["genome_id", "mlst", "split"]].copy()
    out.to_csv(SPLITS_PATH, index=False)
    print(f"Saved {SPLITS_PATH} ({len(out)} rows) — overwritten")


if __name__ == "__main__":
    main()
