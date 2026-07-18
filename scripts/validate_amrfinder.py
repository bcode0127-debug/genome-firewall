#!/usr/bin/env python3
"""Validate AMRFinderPlus gene calls against BV-BRC spgene.tab AR annotations
for the same 20-genome subset.

AMRFinderPlus rows are restricted to Subtype == "AMR" (acquired-gene calls;
POINT / POINT_DISRUPT mutation rows are excluded since spgene has no
point-mutation equivalent to compare against). Gene symbols from both
sources are normalized to a comparable "root" token (lowercased, allele/
suffix numbering stripped) before set comparison, since AMRFinderPlus and
PATRIC/CARD/VFDB gene naming conventions differ.
"""
import csv
import re

ROOT = "/Users/pranayp/Desktop/hacknation"
SUBSET = f"{ROOT}/data/interim/subset20.txt"
AMR_DIR = f"{ROOT}/data/interim/amrfinder"
SPGENE_DIR = f"{ROOT}/data/interim/spgene"
LABELS = f"{ROOT}/data/interim/cohort_labels.available.csv"

CARBAPENEMASE_FAMILIES = ["KPC", "NDM", "OXA", "CTX-M", "CTXM", "SHV", "TEM"]


def normalize(symbol):
    """Strip allele/point-mutation suffixes, the 'bla' beta-lactamase
    prefix, and punctuation; lowercase; drop trailing allele numbers so
    e.g. AMRFinderPlus 'blaKPC-2' and PATRIC 'KPC family' both reduce to
    the family-level token 'kpc'.
    """
    s = symbol.strip()
    # drop point-mutation / allele suffixes like _D87N, _1, -28
    s = re.split(r"[_]", s)[0]
    s = re.sub(r"(?i)^bla", "", s)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    s = s.lower()
    s = re.sub(r"\d+$", "", s)
    return s


def load_amrfinder(genome_id):
    path = f"{AMR_DIR}/{genome_id}.tsv"
    genes_raw = set()
    genes_norm = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("Subtype") != "AMR":
                continue
            sym = row["Element symbol"]
            genes_raw.add(sym)
            genes_norm.add(normalize(sym))
    return genes_raw, genes_norm


ARROW_TAIL_RE = re.compile(r"=>\s*(.+)$")


def family_token_from_product(product):
    """spgene.tab often leaves the 'gene' column blank; PATRIC's K-mer
    annotations instead name the AR determinant in 'product' using a
    controlled "<description> => <family/gene name>[, detail]" pattern,
    e.g. "Class A beta-lactamase (EC 3.5.2.6) => KPC family,
    carbapenem-hydrolyzing". Extracting only the segment right after "=>"
    (up to the first comma, with a trailing " family" stripped) recovers
    the actual gene/family symbol without pulling in generic descriptive
    English words from the rest of the sentence.
    """
    m = ARROW_TAIL_RE.search(product)
    if not m:
        return None
    tail = m.group(1).split(",")[0].strip()
    tail = re.sub(r"\s+family$", "", tail, flags=re.I)
    return tail or None


def load_spgene(genome_id):
    path = f"{SPGENE_DIR}/{genome_id}.spgene.tab"
    genes_raw = set()
    genes_norm = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("property") != "Antibiotic Resistance":
                continue
            sym = row.get("gene", "").strip()
            product = row.get("product", "").strip()
            token = family_token_from_product(product)

            if sym and sym != "-":
                genes_raw.add(sym)
                genes_norm.add(normalize(sym))
            elif token:
                genes_raw.add(token)
                genes_norm.add(normalize(token))
            else:
                genes_raw.add(product[:60])
    return genes_raw, genes_norm


def load_meropenem_labels():
    labels = {}
    with open(LABELS, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["genome_id"]] = row["meropenem"]
    return labels


def main():
    with open(SUBSET) as f:
        ids = [line.strip() for line in f if line.strip()]

    labels = load_meropenem_labels()

    total_both = total_amr_only = total_spgene_only = 0
    per_genome = []

    for gid in ids:
        amr_raw, amr_norm = load_amrfinder(gid)
        sp_raw, sp_norm = load_spgene(gid)

        both = amr_norm & sp_norm
        amr_only = amr_norm - sp_norm
        sp_only = sp_norm - amr_norm

        total_both += len(both)
        total_amr_only += len(amr_only)
        total_spgene_only += len(sp_only)

        carb_hits = sorted(
            sym for sym in amr_raw
            if any(fam.replace("-", "").lower() in normalize(sym) for fam in CARBAPENEMASE_FAMILIES)
        )

        per_genome.append({
            "genome_id": gid,
            "meropenem": labels.get(gid, "?"),
            "amr_raw": sorted(amr_raw),
            "sp_raw": sorted(sp_raw),
            "both": sorted(both),
            "amr_only": sorted(amr_only),
            "sp_only": sorted(sp_only),
            "carb_hits": carb_hits,
        })

    print("=" * 100)
    print("PER-GENOME COMPARISON (AMRFinderPlus acquired-gene calls vs spgene 'Antibiotic Resistance' rows)")
    print("=" * 100)
    for pg in per_genome:
        print(f"\n{pg['genome_id']}  (meropenem={pg['meropenem']})")
        print(f"  AMRFinderPlus AMR genes ({len(pg['amr_raw'])}): {', '.join(pg['amr_raw']) or '(none)'}")
        print(f"  spgene AR genes         ({len(pg['sp_raw'])}): {', '.join(pg['sp_raw']) or '(none)'}")
        print(f"  found by BOTH   ({len(pg['both'])}): {', '.join(pg['both']) or '(none)'}")
        print(f"  AMRFinderPlus ONLY ({len(pg['amr_only'])}): {', '.join(pg['amr_only']) or '(none)'}")
        print(f"  spgene ONLY        ({len(pg['sp_only'])}): {', '.join(pg['sp_only']) or '(none)'}")

    print("\n" + "=" * 100)
    print("OVERALL AGREEMENT")
    print("=" * 100)
    total_calls = total_both + total_amr_only + total_spgene_only
    agreement_pct = (100.0 * total_both / total_calls) if total_calls else 0.0
    print(f"genes found by both:        {total_both}")
    print(f"found only by AMRFinderPlus: {total_amr_only}")
    print(f"found only in spgene:        {total_spgene_only}")
    print(f"overall agreement % (both / union of normalized gene calls): {agreement_pct:.1f}%")

    print("\n" + "=" * 100)
    print("CARBAPENEMASE / ESBL FAMILY CHECK (KPC/NDM/OXA/CTX-M/SHV/TEM) ON THE 10 MEROPENEM-RESISTANT GENOMES")
    print("=" * 100)
    resistant = [pg for pg in per_genome if pg["meropenem"] == "Resistant"]
    n_with_hit = 0
    for pg in resistant:
        hit = "YES" if pg["carb_hits"] else "NO"
        if pg["carb_hits"]:
            n_with_hit += 1
        print(f"  {pg['genome_id']}: {hit}  -> {', '.join(pg['carb_hits']) or '(none)'}")
    print(f"\n{n_with_hit} / {len(resistant)} meropenem-resistant genomes have an AMRFinderPlus call in "
          f"KPC/NDM/OXA/CTX-M/SHV/TEM")


if __name__ == "__main__":
    main()
