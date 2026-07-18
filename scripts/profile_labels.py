import re
import sys

import pandas as pd

RAW_PATH = "data/raw/amr_labels.txt"

# Candidate header names to search for, in priority order. The BV-BRC export
# does not ship a "species" column, so species is derived from genome_name.
PHENOTYPE_CANDIDATES = ["resistant_phenotype", "phenotype"]
LAB_METHOD_CANDIDATES = ["laboratory_typing_method", "lab_typing_method"]
REQUIRED_BASE_COLUMNS = ["genome_id", "genome_name", "antibiotic"]


def find_column(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    print(f"ERROR: could not find a {label} column. Looked for {candidates} "
          f"in headers: {list(columns)}")
    sys.exit(1)


def derive_species(genome_name):
    # Strip surrounding quotes/whitespace, collapse internal whitespace,
    # take the first two tokens as the genus + species epithet.
    cleaned = genome_name.strip().strip('"').strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = cleaned.split(" ")
    return " ".join(tokens[:2]) if len(tokens) >= 2 else cleaned


def main():
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False)

    print("COLUMN NAMES:")
    print(list(df.columns))
    print()

    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        print(f"ERROR: missing required base columns: {missing_base}")
        sys.exit(1)

    phenotype_col = find_column(df.columns, PHENOTYPE_CANDIDATES, "phenotype")
    lab_method_col = find_column(df.columns, LAB_METHOD_CANDIDATES, "lab-typing-method")
    print(f"Using phenotype column: {phenotype_col!r}")
    print(f"Using lab-typing-method column: {lab_method_col!r}")
    if "species" not in df.columns:
        print("No 'species' column present — deriving species from 'genome_name' "
              "(genus + species epithet, first two tokens).")
    print()

    before = len(df)
    df = df[df[lab_method_col] != "Computational Prediction"]
    print(f"Dropped {before - len(df)} rows with lab typing method == "
          f"'Computational Prediction' ({before} -> {len(df)})")

    before = len(df)
    df = df[df[phenotype_col].isin(["Resistant", "Susceptible"])]
    print(f"Kept only phenotype in {{Resistant, Susceptible}} "
          f"({before} -> {len(df)})")
    print()

    df["species"] = df["genome_name"].astype(str).map(derive_species)

    grouped = df.groupby(["species", "antibiotic"]).agg(
        row_count=("genome_id", "size"),
        genome_count=("genome_id", "nunique"),
        resistant_count=(phenotype_col, lambda s: (s == "Resistant").sum()),
    ).reset_index()
    grouped["resistant_fraction"] = grouped["resistant_count"] / grouped["row_count"]

    filtered = grouped[
        (grouped["resistant_fraction"] >= 0.15) & (grouped["resistant_fraction"] <= 0.85)
    ]
    top25 = filtered.sort_values("genome_count", ascending=False).head(25)

    print("TOP 25 SPECIES-ANTIBIOTIC PAIRS BY GENOME COUNT "
          "(resistant fraction in [0.15, 0.85]):")
    print(
        top25[["species", "antibiotic", "row_count", "genome_count", "resistant_fraction"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
