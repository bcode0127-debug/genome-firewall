import re

import pandas as pd

RAW_PATH = "data/raw/amr_labels.txt"
PHENOTYPE_COL = "resistant_phenotype"
LAB_METHOD_COL = "laboratory_typing_method"


def derive_species(genome_name):
    cleaned = genome_name.strip().strip('"').strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = cleaned.split(" ")
    return " ".join(tokens[:2]) if len(tokens) >= 2 else cleaned


def main():
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False)

    # Same base filters as profile_labels.py, to reproduce the 656,158-row set.
    df = df[df[LAB_METHOD_COL] != "Computational Prediction"]
    filtered = df[df[PHENOTYPE_COL].isin(["Resistant", "Susceptible"])].copy()
    filtered["species"] = filtered["genome_name"].astype(str).map(derive_species)

    print(f"Filtered row count: {len(filtered)}")
    print()

    # 1. NaN laboratory_typing_method count within the filtered set.
    nan_count = filtered[LAB_METHOD_COL].isna().sum()
    pct = 100 * nan_count / len(filtered)
    print("=== 1. laboratory_typing_method NaN count (of filtered rows) ===")
    print(f"count: {nan_count}")
    print(f"percent: {pct:.4f}%")
    print()

    # 2. Every antibiotic for each of the two species, no fraction filter.
    species_list = ["Klebsiella pneumoniae", "Escherichia coli"]
    species_tables = {}
    for sp in species_list:
        sub = filtered[filtered["species"] == sp]
        grouped = sub.groupby("antibiotic").agg(
            genome_count=("genome_id", "nunique"),
            row_count=("genome_id", "size"),
            resistant_count=(PHENOTYPE_COL, lambda s: (s == "Resistant").sum()),
        ).reset_index()
        grouped["resistant_fraction"] = grouped["resistant_count"] / grouped["row_count"]
        grouped = grouped.sort_values("genome_count", ascending=False).reset_index(drop=True)
        species_tables[sp] = grouped

        print(f"=== 2. {sp} — every antibiotic, sorted by genome count desc ===")
        print(
            grouped[["antibiotic", "genome_count", "resistant_fraction"]]
            .to_string(index=False)
        )
        print()

    # 3. Top-5 antibiotics per species: co-occurrence with >=1 other top-5 drug.
    for sp in species_list:
        sub = filtered[filtered["species"] == sp]
        top5 = species_tables[sp].head(5)["antibiotic"].tolist()

        genome_sets = {
            drug: set(sub.loc[sub["antibiotic"] == drug, "genome_id"])
            for drug in top5
        }

        print(f"=== 3. {sp} — top 5 antibiotics: genomes with BOTH that drug AND "
              f">=1 other top-5 drug ===")
        for drug in top5:
            others_union = set()
            for other in top5:
                if other != drug:
                    others_union |= genome_sets[other]
            both_count = len(genome_sets[drug] & others_union)
            print(f"{drug}: {both_count} (of {len(genome_sets[drug])} genomes with this drug's label)")
        print()


if __name__ == "__main__":
    main()
