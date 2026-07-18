import re

import pandas as pd

META_PATH = "data/raw/genome_metadata.txt"
GENOME_LIST_PATH = "data/interim/genome_list.available.txt"

GROUPING_KEYWORDS = ["mlst", "sequence type", "strain", "clade", "lineage",
                     "serotype", "serovar", "biovar"]
MLST_COL = "mlst"
LARGE_GROUP_MIN = 20


def is_blank(series):
    s = series.astype("object")
    return s.map(
        lambda v: (v is None) or (isinstance(v, float) and pd.isna(v))
        or (str(v).strip() == "")
    )


def main():
    with open(GENOME_LIST_PATH) as f:
        target_ids = {line.strip() for line in f if line.strip()}
    print(f"target genome ids: {len(target_ids)}")

    meta = pd.read_csv(META_PATH, sep="\t", low_memory=False, dtype={"genome_id": str})
    print(f"genome_metadata.txt rows: {len(meta)}")
    print()

    # === 3. Subset to our 3,342 ids ===
    sub = meta[meta["genome_id"].isin(target_ids)].copy()
    matched_ids = set(sub["genome_id"])
    print("=== 3. Subset match ===")
    print(f"matched: {len(sub)} rows / {len(matched_ids)} distinct genome_ids "
          f"(of {len(target_ids)} target ids)")
    unmatched = target_ids - matched_ids
    print(f"unmatched target ids: {len(unmatched)}")
    print()

    # === 4. Grouping-candidate columns ===
    candidate_cols = []
    for col in meta.columns:
        cl = col.lower()
        if any(kw in cl for kw in GROUPING_KEYWORDS):
            candidate_cols.append(col)
    print(f"=== 4. Grouping-candidate columns found: {candidate_cols} ===")
    print()

    for col in candidate_cols:
        blank_rate = is_blank(sub[col]).mean()
        distinct = sub.loc[~is_blank(sub[col]), col].nunique()
        top10 = sub.loc[~is_blank(sub[col]), col].value_counts().head(10)
        print(f"--- {col} ---")
        print(f"blank rate (on our 3,342): {blank_rate:.4f}")
        print(f"distinct non-blank values: {distinct}")
        print("10 largest groups:")
        print(top10.to_string())
        print()

    # === 5. MLST-specific breakdown ===
    print(f"=== 5. MLST column ({MLST_COL!r}) group-size breakdown ===")
    if MLST_COL not in sub.columns:
        print(f"no column named {MLST_COL!r} present")
    else:
        non_blank = sub.loc[~is_blank(sub[MLST_COL])]
        counts = non_blank[MLST_COL].value_counts()
        n_large_groups = (counts >= LARGE_GROUP_MIN).sum()
        genomes_in_large_groups = counts[counts >= LARGE_GROUP_MIN].sum()
        n_singleton_groups = (counts == 1).sum()
        print(f"non-blank mlst rows: {len(non_blank)}")
        print(f"distinct mlst values: {len(counts)}")
        print(f"groups with >=20 members: {n_large_groups} groups, "
              f"covering {genomes_in_large_groups} genomes")
        print(f"singleton groups (exactly 1 member): {n_singleton_groups} groups, "
              f"{n_singleton_groups} genomes")


if __name__ == "__main__":
    main()
