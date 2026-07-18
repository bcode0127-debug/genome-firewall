#!/usr/bin/env python3
"""Stratified 200-genome sample for MLST-as-homology-proxy validation.

Strata (fixed seed 42, sampled from data/features/splits.csv restricted to
the train+test pool -- calibration is excluded so every stratum can freely
mix train/test):
  - >=40 genomes from ST258
  - >=40 genomes from ST307
  - 40 genomes from 40 distinct singleton STs (STs with exactly 1 genome
    in the train+test pool)
  - 80 genomes spread across mid-size STs (3-15 genomes in the pool),
    capped per-ST so no single mid-size ST dominates the spread

Total: 200. Genomes with a blank mlst field are excluded (not a real ST).
Writes data/interim/subset200_mlst.csv (genome_id, st, split, stratum).
"""
import csv
import random
import collections

ROOT = "/Users/pranayp/Desktop/hacknation"
SPLITS = f"{ROOT}/data/features/splits.csv"
OUT = f"{ROOT}/data/interim/subset200_mlst.csv"
SEED = 42

TARGET_258 = 40
TARGET_307 = 40
N_SINGLETON_STS = 40
N_MIDSIZE_TOTAL = 80
MIDSIZE_MIN, MIDSIZE_MAX = 3, 15
MIDSIZE_PER_ST_CAP = 5  # spread across many mid-size STs, not a few

rng = random.Random(SEED)

pool = []
with open(SPLITS, newline="") as f:
    for row in csv.DictReader(f):
        if row["split"] not in ("train", "test"):
            continue
        if not row["mlst"]:
            continue
        st = row["mlst"].split(".")[-1]
        pool.append({"genome_id": row["genome_id"], "st": st, "split": row["split"]})

by_st = collections.defaultdict(list)
for r in pool:
    by_st[r["st"]].append(r)

selected = []


def take(records, n, stratum):
    chosen = rng.sample(records, min(n, len(records)))
    for c in chosen:
        selected.append({**c, "stratum": stratum})
    return chosen


take(by_st["258"], TARGET_258, "ST258")
take(by_st["307"], TARGET_307, "ST307")

singleton_sts = [st for st, recs in by_st.items() if len(recs) == 1 and st not in ("258", "307")]
rng.shuffle(singleton_sts)
chosen_singleton_sts = singleton_sts[:N_SINGLETON_STS]
for st in chosen_singleton_sts:
    take(by_st[st], 1, "singleton")

midsize_sts = [
    st for st, recs in by_st.items()
    if MIDSIZE_MIN <= len(recs) <= MIDSIZE_MAX and st not in ("258", "307")
]
rng.shuffle(midsize_sts)
n_mid = 0
for st in midsize_sts:
    if n_mid >= N_MIDSIZE_TOTAL:
        break
    take_n = min(MIDSIZE_PER_ST_CAP, len(by_st[st]), N_MIDSIZE_TOTAL - n_mid)
    take(by_st[st], take_n, "midsize")
    n_mid += take_n

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["genome_id", "st", "split", "stratum"])
    w.writeheader()
    for r in selected:
        w.writerow(r)

print(f"total selected: {len(selected)}")
strata_counts = collections.Counter(r["stratum"] for r in selected)
print("by stratum:", dict(strata_counts))
split_counts = collections.Counter(r["split"] for r in selected)
print("by split:", dict(split_counts))
st_counts = collections.Counter(r["st"] for r in selected)
print(f"distinct STs represented: {len(st_counts)}")
print(f"-> {OUT}")
