import glob
import json
import os

import pandas as pd

SPGENE_DIR = "data/interim/spgene"
GENOME_LIST_PATH = "data/interim/genome_list.available.txt"
COHORT_PATH = "data/interim/cohort_labels.available.csv"
MATRIX_PATH = "data/features/matrix.parquet"
VOCAB_PATH = "data/features/vocab.json"
TRAIN_TABLE_PATH = "data/features/train_table.parquet"

RESISTANCE_PROPERTY = "Antibiotic Resistance"
GENE_BLANK_THRESHOLD = 0.20
GENE_MARKERS = ["KPC", "NDM", "OXA", "CTX-M", "SHV", "TEM", "VIM", "IMP"]
SOURCE_BUCKETS = ["CARD", "ARDB", "NDARO", "PATRIC"]


def is_blank(series):
    s = series.astype("object")
    return s.map(
        lambda v: (v is None) or (isinstance(v, float) and pd.isna(v))
        or (str(v).strip() == "")
    )


def main():
    os.makedirs("data/features", exist_ok=True)

    with open(GENOME_LIST_PATH) as f:
        all_genome_ids = [line.strip() for line in f if line.strip()]
    print(f"Target genome universe: {len(all_genome_ids)} ids (from {GENOME_LIST_PATH})")

    files = sorted(glob.glob(os.path.join(SPGENE_DIR, "*.spgene.tab")))
    print(f"Loading {len(files)} .spgene.tab files...")

    frames = []
    for path in files:
        df = pd.read_csv(path, sep="\t", low_memory=False, dtype={"genome_id": str})
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(all_rows)}")

    # === 1. Filter to Antibiotic Resistance rows ===
    ar = all_rows[all_rows["property"] == RESISTANCE_PROPERTY].copy()
    print(f"Antibiotic Resistance rows: {len(ar)}")
    print()

    # === 2. Pick the feature key ===
    print("=== 2. Feature key selection ===")
    for col in ["gene", "product", "source_id"]:
        rate = is_blank(ar[col]).mean()
        print(f"blank rate on AR rows — {col}: {rate:.4f}")

    gene_blank_rate = is_blank(ar["gene"]).mean()
    if gene_blank_rate < GENE_BLANK_THRESHOLD:
        rule = "gene"
        print(f"RULE FIRED: gene blank rate {gene_blank_rate:.4f} < "
              f"{GENE_BLANK_THRESHOLD} threshold -> key = gene")
        ar["key_raw"] = ar["gene"]
    else:
        rule = "gene-else-product"
        print(f"RULE FIRED: gene blank rate {gene_blank_rate:.4f} >= "
              f"{GENE_BLANK_THRESHOLD} threshold -> key = gene where present, "
              f"else product")
        gene_col = ar["gene"].astype("object")
        product_col = ar["product"].astype("object")
        gene_blank_mask = is_blank(ar["gene"])
        ar["key_raw"] = gene_col.where(~gene_blank_mask, product_col)
    print()

    # === 3. Normalize the key ===
    ar = ar[~is_blank(ar["key_raw"])].copy()
    keys_before_split = ar["key_raw"].nunique()

    def split_norm(v):
        parts = str(v).split(",")
        return [p.strip().lower() for p in parts if p.strip() != ""]

    ar["key_list"] = ar["key_raw"].map(split_norm)
    exploded = ar.explode("key_list").rename(columns={"key_list": "key"})
    exploded = exploded[exploded["key"].notna() & (exploded["key"] != "")]

    keys_after_split = exploded["key"].nunique()
    print("=== 3. Key normalization ===")
    print(f"distinct raw keys (pre-split): {keys_before_split}")
    print(f"distinct normalized keys (post-split, lowercased): {keys_after_split}")
    print(f"splitting created {keys_after_split - keys_before_split} additional "
          f"distinct keys")
    print()

    # === 4. Build binary matrix over all target genomes ===
    exploded["genome_id"] = exploded["genome_id"].astype(str)
    present = exploded[["genome_id", "key"]].drop_duplicates()

    vocab = sorted(present["key"].unique().tolist())
    print(f"Feature vocabulary size: {len(vocab)}")

    present = present.assign(_hit=1)
    wide = present.pivot_table(
        index="genome_id", columns="key", values="_hit",
        aggfunc="max", fill_value=0,
    )
    matrix = wide.reindex(index=all_genome_ids, columns=vocab, fill_value=0).astype("int8")
    matrix.index.name = "genome_id"

    matrix_out = matrix.reset_index()
    matrix_out.to_parquet(MATRIX_PATH, index=False)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved {MATRIX_PATH} (shape {matrix.shape})")
    print(f"Saved {VOCAB_PATH} ({len(vocab)} columns, alphabetically ordered)")
    print()

    # === 5. Join to cohort labels ===
    cohort = pd.read_csv(COHORT_PATH, dtype={"genome_id": str})
    train_table = cohort.merge(matrix_out, on="genome_id", how="left")
    train_table.to_parquet(TRAIN_TABLE_PATH, index=False)
    print(f"Saved {TRAIN_TABLE_PATH} (shape {train_table.shape})")
    print()

    # === 6. Diagnostics ===
    print("=== 6. Diagnostics ===")
    n_genomes, n_features = matrix.shape
    total_cells = n_genomes * n_features
    ones = int(matrix.values.sum())
    sparsity = 1 - (ones / total_cells)
    print(f"matrix shape: {n_genomes} genomes x {n_features} features")
    print(f"overall sparsity: {sparsity:.6f} (density: {ones / total_cells:.6f}, "
          f"{ones} of {total_cells} cells set)")
    print()

    freq = matrix.sum(axis=0) / n_genomes
    buckets = {
        ">95%": (freq > 0.95).sum(),
        "70-95%": ((freq >= 0.70) & (freq <= 0.95)).sum(),
        "30-70%": ((freq >= 0.30) & (freq < 0.70)).sum(),
        "5-30%": ((freq >= 0.05) & (freq < 0.30)).sum(),
        "<5%": (freq < 0.05).sum(),
    }
    counts = matrix.sum(axis=0)
    exactly_one = (counts == 1).sum()

    print("genome-frequency buckets:")
    for label, count in buckets.items():
        print(f"  {label:<8}: {count}")
    print(f"  exactly 1 genome (singleton features): {exactly_one}")
    print()

    band_cols = freq[(freq >= 0.05) & (freq < 0.70)].sort_values(ascending=False)
    print(f"columns in the 5-70% band ({len(band_cols)}):")
    for col, f in band_cols.items():
        print(f"  {col:<30}{f:.4f}")
    print()

    print("marker gene-family search (case-insensitive substring in column names):")
    for marker in GENE_MARKERS:
        matches = [c for c in vocab if marker.lower() in c.lower()]
        if not matches:
            print(f"  {marker:<8}: NOT FOUND")
        else:
            for m in matches:
                pct = freq[m] * 100
                print(f"  {marker:<8}: found column '{m}' — {pct:.2f}% of genomes")
    print()

    print("AR row counts by source:")
    source_counts = ar["source"].value_counts(dropna=False)
    for bucket in SOURCE_BUCKETS:
        cnt = source_counts.get(bucket, 0)
        print(f"  {bucket:<10}: {cnt}")
    other_sources = set(source_counts.index) - set(SOURCE_BUCKETS)
    if other_sources:
        print("  other sources present in AR rows:")
        for s in sorted(other_sources, key=lambda x: str(x)):
            print(f"    {s}: {source_counts[s]}")


if __name__ == "__main__":
    main()
