import re

import pandas as pd

RAW_PATH = "data/raw/amr_labels.txt"
PHENOTYPE_COL = "resistant_phenotype"
LAB_METHOD_COL = "laboratory_typing_method"

TARGET_SPECIES = "Klebsiella pneumoniae"
TARGET_DRUGS = ["meropenem", "ceftazidime", "gentamicin"]


def derive_species(genome_name):
    if pd.isna(genome_name):
        return ""
    cleaned = str(genome_name).strip().strip('"').strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = cleaned.split(" ")
    return " ".join(tokens[:2]) if len(tokens) >= 2 else cleaned


def filled(series):
    """Non-null AND not an empty/whitespace-only string."""
    s = series.astype("object")
    notnull = series.notna()
    nonempty = s.map(lambda v: str(v).strip() != "" if v is not None else False)
    return notnull & nonempty


def main():
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False)

    # Reproduce the 656,158-row filtered set.
    df = df[df[LAB_METHOD_COL] != "Computational Prediction"]
    filt = df[df[PHENOTYPE_COL].isin(["Resistant", "Susceptible"])].copy()
    print(f"Filtered rows: {len(filt)}")
    print()

    pair = ["genome_id", "antibiotic"]

    # --- 1. Duplicate (genome_id, antibiotic) pairs and conflicts ---
    grp = filt.groupby(pair)
    rows_per_pair = grp.size()
    distinct_pheno = grp[PHENOTYPE_COL].nunique()

    total_pairs = len(rows_per_pair)
    dup_pairs = (rows_per_pair > 1).sum()
    conflict_pairs = (distinct_pheno > 1).sum()

    print("=== 1. Duplicate & conflicting (genome_id, antibiotic) pairs ===")
    print(f"distinct pairs: {total_pairs}")
    print(f"pairs with >1 row (duplicated): {dup_pairs}")
    print(f"  of those, extra rows collapsed if deduped: {len(filt) - total_pairs}")
    print(f"pairs that CONFLICT (>1 distinct resistant_phenotype): {conflict_pairs}")
    print(f"  conflicts as % of duplicated pairs: "
          f"{100 * conflict_pairs / dup_pairs:.2f}%")
    print()

    # Identify the conflicting pairs and pull their rows.
    conflict_keys = distinct_pheno[distinct_pheno > 1].index
    conflict_mask = filt.set_index(pair).index.isin(conflict_keys)
    conflict_rows = filt[conflict_mask].copy()
    conflict_rows["method_is_nan"] = conflict_rows[LAB_METHOD_COL].isna()

    # --- 2. Conflicting pairs: method-presence structure within the pair ---
    print("=== 2. Conflicting pairs — is method presence split within the pair? ===")
    per_pair_method = conflict_rows.groupby(pair)["method_is_nan"].agg(
        n_rows="size", n_nan="sum"
    )
    per_pair_method["n_real"] = per_pair_method["n_rows"] - per_pair_method["n_nan"]

    all_real = ((per_pair_method["n_nan"] == 0)).sum()
    all_nan = ((per_pair_method["n_real"] == 0)).sum()
    mixed = ((per_pair_method["n_nan"] > 0) & (per_pair_method["n_real"] > 0)).sum()
    print(f"conflicting pairs total: {len(per_pair_method)}")
    print(f"  all rows have a real method:        {all_real}")
    print(f"  all rows have NaN method:           {all_nan}")
    print(f"  MIXED (>=1 real AND >=1 NaN):        {mixed}")
    print()
    print("Row-level cross-tab within conflicting pairs "
          "(phenotype x method-is-NaN):")
    ct = pd.crosstab(conflict_rows[PHENOTYPE_COL], conflict_rows["method_is_nan"])
    ct.columns = ["method_present" if c is False else "method_NaN" for c in ct.columns]
    print(ct.to_string())
    print()

    # --- 3. Fill rates: NaN-method rows vs non-NaN-method rows ---
    print("=== 3. Fill rates by method presence (filled = non-null, non-empty) ===")
    nan_rows = filt[filt[LAB_METHOD_COL].isna()]
    real_rows = filt[filt[LAB_METHOD_COL].notna()]
    print(f"NaN-method rows: {len(nan_rows)}   non-NaN-method rows: {len(real_rows)}")
    cols = ["measurement_value", "testing_standard", "vendor", "source"]
    header = f"{'column':<20}{'NaN-method fill':>18}{'non-NaN fill':>16}"
    print(header)
    for c in cols:
        nan_fill = filled(nan_rows[c]).mean() if len(nan_rows) else float("nan")
        real_fill = filled(real_rows[c]).mean() if len(real_rows) else float("nan")
        print(f"{c:<20}{nan_fill:>17.4f}{real_fill:>16.4f}")
    print()

    # --- 4. Did the cohort pivot silently collapse duplicate pairs? ---
    print("=== 4. Cohort pivot: were duplicate pairs silently collapsed? ===")
    filt["species"] = filt["genome_name"].map(derive_species)
    coh = filt[filt["species"] == TARGET_SPECIES].copy()
    coh["antibiotic_norm"] = coh["antibiotic"].astype(str).str.strip().str.lower()
    coh = coh[coh["antibiotic_norm"].isin(TARGET_DRUGS)]

    cpair = ["genome_id", "antibiotic_norm"]
    cgrp = coh.groupby(cpair)
    c_rows_per_pair = cgrp.size()
    c_distinct = cgrp[PHENOTYPE_COL].nunique()

    c_total_pairs = len(c_rows_per_pair)
    c_dup_pairs = (c_rows_per_pair > 1).sum()
    c_conflict = (c_distinct > 1).sum()
    c_collapsed = len(coh) - c_total_pairs

    print(f"cohort candidate rows (K.pneu x 3 drugs, R/S): {len(coh)}")
    print(f"distinct (genome_id, drug) pairs: {c_total_pairs}")
    print(f"pairs with >1 row (collapsed by pivot 'first'): {c_dup_pairs}")
    print(f"extra rows silently dropped by pivot: {c_collapsed}")
    print(f"of collapsed pairs, CONFLICTING phenotype: {c_conflict}")

    # How many of the 3,584 all-3 genomes touched a collapsed/conflicting pair.
    pivot = coh.pivot_table(index="genome_id", columns="antibiotic_norm",
                            values=PHENOTYPE_COL, aggfunc="first")
    pivot = pivot.reindex(columns=TARGET_DRUGS)
    all3 = pivot[pivot.notna().sum(axis=1) == 3]
    print(f"all-3-label genomes (pivot): {len(all3)}")

    dup_pair_keys = c_rows_per_pair[c_rows_per_pair > 1].index
    dup_genomes = {g for (g, _d) in dup_pair_keys}
    conflict_pair_keys = c_distinct[c_distinct > 1].index
    conflict_genomes = {g for (g, _d) in conflict_pair_keys}
    print(f"  all-3 genomes touching a duplicated pair: "
          f"{len(set(all3.index) & dup_genomes)}")
    print(f"  all-3 genomes touching a CONFLICTING pair: "
          f"{len(set(all3.index) & conflict_genomes)}")


if __name__ == "__main__":
    main()
