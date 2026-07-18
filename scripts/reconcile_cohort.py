import glob
import os

import pandas as pd

COHORT_PATH = "data/interim/cohort_labels.csv"
SPGENE_DIR = "data/interim/spgene"
AVAIL_CSV = "data/interim/cohort_labels.available.csv"
AVAIL_LIST = "data/interim/genome_list.available.txt"
DRUGS = ["meropenem", "ceftazidime", "gentamicin"]


def main():
    cohort = pd.read_csv(COHORT_PATH, dtype={"genome_id": str})
    print(f"cohort_labels.csv rows: {len(cohort)}")

    # Actual annotation files present -> genome ids.
    files = glob.glob(os.path.join(SPGENE_DIR, "*.spgene.tab"))
    present_ids = {os.path.basename(f)[: -len(".spgene.tab")] for f in files}
    print(f".spgene.tab files present: {len(present_ids)}")

    cohort_ids = set(cohort["genome_id"])
    available_ids = cohort_ids & present_ids
    dropped_ids = cohort_ids - present_ids
    print(f"in both (available): {len(available_ids)}")
    print(f"dropped (in cohort, no annotation): {len(dropped_ids)}")
    print()

    available = cohort[cohort["genome_id"].isin(available_ids)].copy()
    dropped = cohort[cohort["genome_id"].isin(dropped_ids)].copy()

    # Write the two new files (do not touch originals).
    available.to_csv(AVAIL_CSV, index=False)
    with open(AVAIL_LIST, "w") as f:
        for gid in available["genome_id"]:
            f.write(f"{gid}\n")
    print(f"wrote {AVAIL_CSV} ({len(available)} rows)")
    print(f"wrote {AVAIL_LIST} ({len(available)} ids)")
    print()

    def rfrac(df, drug):
        return (df[drug] == "Resistant").mean() if len(df) else float("nan")

    print("=== Resistant fraction per drug, three ways ===")
    print(f"{'drug':<14}{'full (3584)':>14}{'available (3342)':>20}{'dropped (242)':>16}")
    for drug in DRUGS:
        print(f"{drug:<14}{rfrac(cohort, drug):>14.4f}"
              f"{rfrac(available, drug):>20.4f}{rfrac(dropped, drug):>16.4f}")
    print()
    print(f"(row counts — full: {len(cohort)}, available: {len(available)}, "
          f"dropped: {len(dropped)})")


if __name__ == "__main__":
    main()
