import glob
import os
import random

import pandas as pd

SPGENE_DIR = "data/interim/spgene"
SAMPLE_SIZE = 50
SEED = 42

RESISTANCE_PROPERTY = "Antibiotic Resistance"


def blank_rate(series):
    s = series.astype("object")
    is_blank = s.map(
        lambda v: (v is None) or (isinstance(v, float) and pd.isna(v))
        or (str(v).strip() == "")
    )
    return is_blank.mean()


def main():
    all_files = sorted(glob.glob(os.path.join(SPGENE_DIR, "*.spgene.tab")))
    print(f"total .spgene.tab files available: {len(all_files)}")

    random.seed(SEED)
    sample_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    print(f"sampled {len(sample_files)} files (seed={SEED})")
    print()

    frames = []
    for path in sample_files:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"total rows across sample: {len(data)}")
    print()

    print("=== a) distinct `property` values, with row counts ===")
    print(data["property"].value_counts(dropna=False).to_string())
    print()

    print("=== b) distinct `source` values, with row counts ===")
    print(data["source"].value_counts(dropna=False).to_string())
    print()

    print("=== c) null/blank rate ===")
    cols = ["gene", "product", "source_id", "refseq_locus_tag"]
    for c in cols:
        rate = blank_rate(data[c])
        print(f"{c:<20}{rate:.4f}")
    print()

    print(f'=== d) antibiotic-resistance rows (property == "{RESISTANCE_PROPERTY}") ===')
    ar = data[data["property"] == RESISTANCE_PROPERTY]
    print(f"AR rows: {len(ar)}")
    distinct_genes = ar["gene"].nunique(dropna=True)
    print(f"distinct gene count: {distinct_genes}")
    print()
    print("top 20 most common genes:")
    print(ar["gene"].value_counts(dropna=True).head(20).to_string())


if __name__ == "__main__":
    main()
