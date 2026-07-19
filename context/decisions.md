# Decisions

Append-only log of non-obvious decisions made while building this project. One entry per decision.

## Log

- Backend venv must use Python 3.12 (or 3.11/3.13), not 3.14 — pydantic-core
  has no prebuilt wheel for 3.14 yet and fails to build from source (pyo3
  doesn't support 3.14). Use `python3.12 -m venv venv`.

## Genome Firewall — Architecture Decisions

### ADR-001 — Species and drug panel
Chose: Klebsiella pneumoniae; meropenem, ceftazidime, gentamicin.
Rejected: M. tuberculosis (highest sample count), E. coli (1.7x genomes).
Why: MTB resistance is chromosomal point mutations and MTB is
genetically monomorphic — breaks both gene-presence features and the
grouped split. E. coli's top drug panel is class-imbalanced
(ampicillin) and mechanistically redundant (two blaCTX-M drugs).
K. pneumoniae clears all six constraints: class balance, volume,
acquired-gene mechanisms, strain diversity, near-total drug
co-occurrence, and clinical salience (WHO critical priority).

Cohort: 3,584 genomes with all three labels, of which 3,342
have retrievable annotations. Resistant fractions on the
3,584: meropenem 0.330, ceftazidime 0.729, gentamicin 0.381
— all within the 0.15-0.85 class-balance bound.

### ADR-002 — Label source and provenance
Chose: BV-BRC PATRIC_genomes_AMR.txt, laboratory-measured only.
Rejected: general phenotype fields; predicted phenotypes.
Why: the brief warns general phenotype fields may contain
model-generated predictions. We explicitly filtered
laboratory_typing_method == "Computational Prediction" and verified
0 matches, confirming the export is lab-methods-only. Rows with a
blank laboratory_typing_method (15.6%) are retained. We tested
whether these were model-generated: BV-BRC does serve
XGBoost-predicted phenotypes, but those carry predicted MIC values.
Our blank-method rows fill measurement_value only 4.7% of the time
versus 38.7% for method-recorded rows, and are similarly sparse on
vendor and testing standard. They are under-documented laboratory
records, not predictions. We note the limitation explicitly rather
than claiming full metadata parity.

### ADR-003 — Feature source
Chose: BV-BRC specialty-gene annotations (CARD-sourced) for
the training set; AMRFinderPlus run directly on a 20-genome
subset.
Rejected: AMRFinderPlus across all 3,342 genomes.
Why: full annotation does not fit the event window on CPU.
The subset run establishes the documented, repeatable
FASTA-to-features path the brief requires. The precomputed
annotations supply gene-presence signal at scale from a
curated resistance database, which we state plainly rather
than presenting as AMRFinderPlus output. Rows are filtered
to property == "Antibiotic Resistance" before
featurisation, since the specialty-gene file also contains
virulence factors, drug targets, and transporters. Feature
vocabulary is frozen from training genomes only and
versioned in features/vocab.json so inference column order
cannot drift.

### ADR-004 — Train / calibration / test split
Chose: GroupKFold on MLST sequence type from BV-BRC
genome_metadata. Untyped genomes (230) become singleton groups.
Rejected: Mash sketch + distance clustering (ADR-004 v1);
random row splits.
Why: Mash requires all 3,342 assemblies (~19 GB) which does not
fit the event window; MLST is already curated, covers 93.1% of
the cohort, and yields 23 groups of >=20 members over 2,183
genomes. It is also more interpretable than a distance cutoff:
holding out a named clone such as ST258 lets us state plainly
which lineage the model never saw. Mash remains installed as an
unused fallback.

### ADR-005 — No-call mechanism
Chose: two independent triggers — conformal prediction on the
calibration split, plus a novelty gate.
Rejected: single softmax probability threshold.
Why: conformal targets class-conditional coverage rather than a tuned
cutoff — but the guarantee assumes calibration and test are
exchangeable, and our MLST group split deliberately breaks that, so we
report the MEASURED coverage rather than claiming a distribution-free
guarantee. Measured class-conditional coverage misses nominal on every
drug (meropenem susceptible 0.836, ceftazidime susceptible 0.822,
gentamicin resistant 0.860). The novelty gate (nearest-neighbour
distance in feature space, plus count of AMR elements absent from the
training vocabulary) covers the brief's "unlike the training data"
clause, which a probability threshold cannot express.

### ADR-006 — Drug-target presence gate
Chose: deterministic, non-ML, applied before the model.
Rejected: letting the model infer target relevance.
Why: the brief requires the system never report "likely to work"
from absent resistance markers alone. If the drug's molecular target
is absent, the drug is irrelevant and the result is forced to
no-call regardless of model output.

### ADR-007 — Scope of the language model
Chose: LLM confined to the reporting layer, downstream of the verdict.
Rejected: LLM anywhere in the prediction path.
Why: the medical output must be deterministic, auditable, and
reproducible. The LLM turns a fixed structured verdict into
plain-language text and never influences it. This is a safety
property, stated explicitly in the demo.

### ADR-008 — Duplicate and conflicting labels
Chose: explicit deterministic rule — collapse duplicate
(genome_id, antibiotic) pairs that agree; drop pairs that
conflict; log both counts.
Rejected: pandas pivot_table(aggfunc="first"), which
resolves duplicates implicitly.
Why: the brief requires one final label per genome-antibiotic
pair. Across the full filtered set, 39,446 pairs are
duplicated and 5,790 conflict — 73% of those conflicts are
between two method-recorded rows, reflecting breakpoint
differences between testing standards and years rather than
data error. Within our cohort, 81 duplicates were collapsed
across 59 genomes and none conflicted, so this rule changes
no label here; it is made explicit so the pipeline is
correct by construction rather than by coincidence.

### ADR-009 — Genomes without retrievable annotations
Chose: proceed on the 3,342 genomes with annotations; document
the 242-genome (6.75%) shortfall rather than substitute or impute.
Rejected: back-filling missing genomes from another species pool
or relaxing the all-three-drugs requirement to recover count.
Why: the 242 failures return FTP 550 — the genome directory does
not exist on the server, likely withdrawn or superseded
accessions clustered in the low id range. We verified the drop
does not bias class balance: resistant fractions move from
0.330/0.729/0.381 (full 3,584) to 0.331/0.733/0.380 (available
3,342), with the dropped set only marginally more susceptible to
ceftazidime (0.674). 3,342 remains within the brief's 1,000-3,000
target band. The available cohort is the single source of truth
downstream; the label cohort is retained for lineage.

### ADR-010 — AMRFinderPlus vs. spgene.tab gene-name comparison
Chose: normalize gene symbols to a family-level token (lowercase,
strip the "bla" prefix, strip trailing allele digits, e.g.
"blaKPC-2" and "KPC-3" both -> "kpc") before set comparison, and
for spgene.tab rows with a blank `gene` column, recover the
family/gene name from the `product` column's "<description> =>
<family name>[, detail]" segment rather than mining every word in
the free-text description.
Rejected: exact-string gene matching (fails completely — the two
sources use unrelated naming conventions); mining all words >=3
chars from `product` as fallback tokens (tried first; on a
20-genome smoke test it pulled in generic English words like
"class", "family", "with" as spurious matches, inflating both
"spgene only" and, more importantly, risking false "both" hits).
Why: AMRFinderPlus and PATRIC/CARD/VFDB (feeding spgene.tab) use
different gene-nomenclature conventions and allele granularity, so
comparison must happen at the family-symbol level to be meaningful
at all. About 40% of spgene.tab's "Antibiotic Resistance" rows
across our 20-genome validation subset (data/interim/spgene/) have
an empty `gene` column entirely (ARDB/CARD BLASTP-sourced rows
mostly stay generic and have no recoverable symbol; PATRIC K-mer
rows reliably use the "=>" convention) — naively comparing only the
`gene` column silently zeroed out spgene-side calls for 8/20
genomes in this subset. See scripts/validate_amrfinder.py and
data/interim/validation_report.txt. Even after this fix, raw
agreement is low (~10.6% both / union) because AMRFinderPlus only
reports curated acquired-resistance genes and point mutations,
while spgene's "Antibiotic Resistance" property also tags
intrinsic/core genes present in nearly every genome (efflux pumps
like AcrAB-TolC, porins, RNA polymerase, DNA gyrase) — this is a
genuine scope difference between the two tools, not a bug.

### ADR-011 — MLST is not a perfectly clean homology proxy (ST258/ST437)
Chose: report both a strict threshold (within-ST max / between-ST
min midpoint) and a practical bulk threshold (within-ST p95 /
between-ST p5 midpoint) rather than a single number, and flag that
MLST-based grouping is NOT perfectly consistent with either.
Why: on the 200-genome stratified Mash validation
(scripts/select_mlst_subset200.py, scripts/run_mash.sh,
scripts/analyze_mash.py; see data/results/mash_validation.json),
within-ST and between-ST Mash-distance distributions overlap
(within-ST max 0.00813 > between-ST min 0.00339). The overlap is
not broad noise — 200 of 248 overlapping pairs (81%) are ST258 vs
ST437 specifically, with the remaining spread thinly across 11
other ST-pairs. ST437 is a known single-locus variant of ST258
within the same clonal complex (CG258), so this is biologically
expected, not a data-quality problem. Bulk statistics (mean, p95/
p5) still separate cleanly (within mean 0.0018 vs between mean
0.0145), so MLST remains a good coarse proxy for homology, but
callers relying on ST alone to guarantee no near-duplicate genomes
across a train/test split should not assume closely-related STs
(e.g. same clonal complex) are safely dissimilar.
No cross-split leakage (Mash distance < 0.001) was found in this
subset.
