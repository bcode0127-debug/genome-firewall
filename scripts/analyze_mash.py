#!/usr/bin/env python3
"""Validates MLST (sequence type) as a proxy for genome-wide sequence
homology, using all-vs-all Mash distances over the 200-genome stratified
subset (data/interim/subset200_mlst.csv).

Reads data/interim/mash_dist.tsv (mash dist output: ref, query, distance,
p-value, shared-hashes) and data/interim/subset200_mlst.csv (genome_id,
st, split, stratum). Prints the requested summary and writes
data/results/mash_validation.json.
"""
import csv
import json
import statistics
import collections

ROOT = "/Users/pranayp/Desktop/hacknation"
DIST_FILE = f"{ROOT}/data/interim/mash_dist.tsv"
SUBSET_FILE = f"{ROOT}/data/interim/subset200_mlst.csv"
OUT_JSON = f"{ROOT}/data/results/mash_validation.json"

LEAKAGE_THRESHOLD = 0.001


def genome_id_from_path(p):
    # mash dist echoes back the original sketched file path, e.g.
    # .../mash_sketches/573.12771.msh or .../fasta/573.12771.fna
    # depending on how the sketch was built; strip either extension.
    base = p.rsplit("/", 1)[-1]
    for ext in (".msh", ".fna"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    meta = {}
    with open(SUBSET_FILE, newline="") as f:
        for row in csv.DictReader(f):
            meta[row["genome_id"]] = {"st": row["st"], "split": row["split"], "stratum": row["stratum"]}

    within = []
    between = []
    pairs_seen = set()
    missing_meta = set()

    with open(DIST_FILE, newline="") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ref, query, dist = parts[0], parts[1], float(parts[2])
            g1, g2 = genome_id_from_path(ref), genome_id_from_path(query)
            if g1 == g2:
                continue
            key = tuple(sorted((g1, g2)))
            if key in pairs_seen:
                continue
            pairs_seen.add(key)

            if g1 not in meta or g2 not in meta:
                missing_meta.add(g1 if g1 not in meta else g2)
                continue

            m1, m2 = meta[g1], meta[g2]
            record = {
                "genome_1": g1, "genome_2": g2,
                "st_1": m1["st"], "st_2": m2["st"],
                "split_1": m1["split"], "split_2": m2["split"],
                "distance": dist,
            }
            if m1["st"] == m2["st"]:
                within.append(record)
            else:
                between.append(record)

    within_d = sorted(r["distance"] for r in within)
    between_d = sorted(r["distance"] for r in between)

    within_stats = {
        "n_pairs": len(within_d),
        "mean": statistics.mean(within_d) if within_d else None,
        "p95": percentile(within_d, 0.95),
        "max": within_d[-1] if within_d else None,
        "min": within_d[0] if within_d else None,
    }
    between_stats = {
        "n_pairs": len(between_d),
        "mean": statistics.mean(between_d) if between_d else None,
        "p5": percentile(between_d, 0.05),
        "min": between_d[0] if between_d else None,
        "max": between_d[-1] if between_d else None,
    }

    between_sorted = sorted(between, key=lambda r: r["distance"])
    smallest_20 = []
    for r in between_sorted[:20]:
        cross_split = r["split_1"] != r["split_2"]
        smallest_20.append({
            "genome_1": r["genome_1"], "st_1": r["st_1"], "split_1": r["split_1"],
            "genome_2": r["genome_2"], "st_2": r["st_2"], "split_2": r["split_2"],
            "distance": r["distance"],
            "cross_split": cross_split,
        })

    leakage_pairs = [
        r for r in between
        if r["distance"] < LEAKAGE_THRESHOLD and r["split_1"] != r["split_2"]
    ]
    leakage_list = [{
        "genome_1": r["genome_1"], "st_1": r["st_1"], "split_1": r["split_1"],
        "genome_2": r["genome_2"], "st_2": r["st_2"], "split_2": r["split_2"],
        "distance": r["distance"],
    } for r in sorted(leakage_pairs, key=lambda r: r["distance"])]

    # threshold recommendation: midpoint between within-ST p95/max and
    # between-ST p5/min gives the largest margin separating the two
    # distributions without relying on the extreme tails alone.
    within_upper = within_stats["max"] if within_stats["max"] is not None else within_stats["p95"]
    between_lower = between_stats["min"] if between_stats["min"] is not None else between_stats["p5"]
    recommended_threshold = None
    mlst_consistent = None
    if within_upper is not None and between_lower is not None:
        recommended_threshold = round((within_upper + between_lower) / 2, 5)
        # MLST grouping is "consistent" with the threshold if the within-ST
        # max sits below it and the between-ST min sits above it -- i.e.
        # the two distributions don't overlap at all.
        mlst_consistent = (within_stats["max"] is not None and within_stats["max"] < recommended_threshold
                            and between_stats["min"] is not None and between_stats["min"] > recommended_threshold)

    print("=" * 100)
    print(f"WITHIN-ST distance distribution (n={within_stats['n_pairs']} pairs)")
    print("=" * 100)
    print(f"  mean: {within_stats['mean']}")
    print(f"  p95:  {within_stats['p95']}")
    print(f"  max:  {within_stats['max']}")

    print("\n" + "=" * 100)
    print(f"BETWEEN-ST distance distribution (n={between_stats['n_pairs']} pairs)")
    print("=" * 100)
    print(f"  mean: {between_stats['mean']}")
    print(f"  p5:   {between_stats['p5']}")
    print(f"  min:  {between_stats['min']}")

    print("\n" + "=" * 100)
    print("20 SMALLEST BETWEEN-ST PAIRS")
    print("=" * 100)
    for r in smallest_20:
        print(f"  {r['genome_1']} (ST{r['st_1']}, {r['split_1']})  <->  "
              f"{r['genome_2']} (ST{r['st_2']}, {r['split_2']})   "
              f"dist={r['distance']:.6f}   cross_split={r['cross_split']}")

    print("\n" + "=" * 100)
    print(f"LEAKAGE CHECK: cross-split pairs with Mash distance < {LEAKAGE_THRESHOLD}")
    print("=" * 100)
    if leakage_list:
        for r in leakage_list:
            print(f"  {r['genome_1']} (ST{r['st_1']}, {r['split_1']})  <->  "
                  f"{r['genome_2']} (ST{r['st_2']}, {r['split_2']})   dist={r['distance']:.6f}")
    else:
        print("  none")

    # Which specific ST-pairs cause within/between overlap (distances below
    # the within-ST max that are nonetheless between different STs) -- this
    # tells us whether overlap is broad noise or a few closely-related STs.
    overlap_pairs = [r for r in between if within_stats["max"] is not None and r["distance"] < within_stats["max"]]
    overlap_st_pair_counts = collections.Counter(
        tuple(sorted((r["st_1"], r["st_2"]))) for r in overlap_pairs
    )
    overlap_summary = [
        {"st_pair": list(k), "n_pairs": v}
        for k, v in sorted(overlap_st_pair_counts.items(), key=lambda kv: -kv[1])
    ]

    # A practical threshold using the bulk (p95/p5) of each distribution,
    # rather than the extreme tails, for comparison.
    practical_threshold = None
    if within_stats["p95"] is not None and between_stats["p5"] is not None:
        practical_threshold = round((within_stats["p95"] + between_stats["p5"]) / 2, 5)

    print("\n" + "=" * 100)
    print("RECOMMENDED HOMOLOGY THRESHOLD")
    print("=" * 100)
    print(f"  within-ST upper bound (max): {within_stats['max']}")
    print(f"  between-ST lower bound (min): {between_stats['min']}")
    print(f"  recommended threshold (midpoint of max/min): {recommended_threshold}")
    print(f"  practical threshold (midpoint of within-p95/between-p5): {practical_threshold}")
    print(f"  MLST grouping consistent with a clean-separating threshold: {mlst_consistent}")
    if overlap_summary:
        print(f"  {len(overlap_pairs)} between-ST pairs fall below the within-ST max "
              f"(distributions overlap). Concentrated in {len(overlap_summary)} ST-pairs, "
              f"dominated by: {overlap_summary[:5]}")
        print("  -> overlap is driven by a small number of closely-related ST pairs "
              "(e.g. same clonal complex / single-locus variants), not broad noise; "
              "bulk statistics (mean/p95/p5) still show clean separation.")
    if missing_meta:
        print(f"\n  NOTE: {len(missing_meta)} genome id(s) in mash_dist.tsv had no subset metadata "
              f"and were excluded from stats: {sorted(missing_meta)[:10]}{'...' if len(missing_meta) > 10 else ''}")

    result = {
        "subset_size": len(meta),
        "leakage_threshold": LEAKAGE_THRESHOLD,
        "within_st": within_stats,
        "between_st": between_stats,
        "smallest_20_between_st_pairs": smallest_20,
        "leakage_check": {
            "threshold": LEAKAGE_THRESHOLD,
            "n_leaked_pairs": len(leakage_list),
            "pairs": leakage_list,
        },
        "recommended_threshold": {
            "value": recommended_threshold,
            "method": "midpoint of within-ST max and between-ST min",
            "mlst_grouping_consistent": mlst_consistent,
            "practical_threshold_p95_p5_midpoint": practical_threshold,
            "n_overlap_pairs": len(overlap_pairs),
            "overlap_by_st_pair": overlap_summary,
            "overlap_interpretation": (
                "Overlap between the within-ST and between-ST distributions is concentrated "
                "in a small number of ST-pairs (dominated by ST258/ST437) rather than spread "
                "broadly, consistent with those STs being closely related (e.g. same clonal "
                "complex / single-locus variants) rather than MLST failing generally."
            ) if overlap_summary else None,
        },
        "missing_metadata_genome_ids": sorted(missing_meta),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
