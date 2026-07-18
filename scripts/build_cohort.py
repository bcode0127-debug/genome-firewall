import re

import pandas as pd

RAW_PATH = "data/raw/amr_labels.txt"
PHENOTYPE_COL = "resistant_phenotype"
COHORT_LABELS_PATH = "data/interim/cohort_labels.csv"
GENOME_LIST_PATH = "data/interim/genome_list.txt"

TARGET_SPECIES = "Klebsiella pneumoniae"
TARGET_DRUGS = ["meropenem", "ceftazidime", "gentamicin"]


def derive_species(genome_name):
    if pd.isna(genome_name):
        return ""
    cleaned = str(genome_name).strip().strip('"').strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = cleaned.split(" ")
    return " ".join(tokens[:2]) if len(tokens) >= 2 else cleaned


def main():
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False)

    df["species"] = df["genome_name"].map(derive_species)
    df = df[df["species"] == TARGET_SPECIES]

    df["antibiotic_norm"] = df["antibiotic"].astype(str).str.strip().str.lower()
    df = df[df["antibiotic_norm"].isin(TARGET_DRUGS)]

    df = df[df[PHENOTYPE_COL].isin(["Resistant", "Susceptible"])]

    pivot = df.pivot_table(
        index="genome_id",
        columns="antibiotic_norm",
        values=PHENOTYPE_COL,
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=TARGET_DRUGS)

    label_count = pivot.notna().sum(axis=1)
    total_genomes = len(pivot)
    with_all_3 = (label_count == 3).sum()
    with_exactly_2 = (label_count == 2).sum()
    with_exactly_1 = (label_count == 1).sum()

    print("=== Genome counts ===")
    print(f"total genomes: {total_genomes}")
    print(f"with all 3 labels: {with_all_3}")
    print(f"with exactly 2 labels: {with_exactly_2}")
    print(f"with exactly 1 label: {with_exactly_1}")
    print()

    all3 = pivot[label_count == 3].copy()

    print("=== Resistant fraction per drug (all-3 subset) ===")
    for drug in TARGET_DRUGS:
        resistant_fraction = (all3[drug] == "Resistant").mean()
        print(f"{drug}: {resistant_fraction:.6f}")
    print()

    all3.to_csv(COHORT_LABELS_PATH)
    print(f"Saved {len(all3)} rows to {COHORT_LABELS_PATH}")

    with open(GENOME_LIST_PATH, "w") as f:
        for genome_id in all3.index:
            f.write(f"{genome_id}\n")
    print(f"Saved {len(all3)} genome ids to {GENOME_LIST_PATH}")


if __name__ == "__main__":
    main()
