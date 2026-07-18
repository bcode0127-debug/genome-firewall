#!/usr/bin/env python3
"""Stratified 20-genome subset: 10 meropenem-Resistant, 10 meropenem-Susceptible.

Fixed seed (42) over the available cohort so the selection is reproducible.
Writes one genome_id per line to data/interim/subset20.txt.
"""
import csv
import random

ROOT = "/Users/pranayp/Desktop/hacknation"
SRC = f"{ROOT}/data/interim/cohort_labels.available.csv"
OUT = f"{ROOT}/data/interim/subset20.txt"
SEED = 42
N_PER_CLASS = 10

resistant, susceptible = [], []
with open(SRC, newline="") as f:
    for row in csv.DictReader(f):
        if row["meropenem"] == "Resistant":
            resistant.append(row["genome_id"])
        elif row["meropenem"] == "Susceptible":
            susceptible.append(row["genome_id"])

rng = random.Random(SEED)
rng.shuffle(resistant)
rng.shuffle(susceptible)

subset = sorted(resistant[:N_PER_CLASS]) + sorted(susceptible[:N_PER_CLASS])

with open(OUT, "w") as f:
    for gid in subset:
        f.write(gid + "\n")

print(f"resistant pool: {len(resistant)}, susceptible pool: {len(susceptible)}")
print(f"selected {N_PER_CLASS} resistant + {N_PER_CLASS} susceptible -> {OUT}")
for gid in subset:
    print(gid)
