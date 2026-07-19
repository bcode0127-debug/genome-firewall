# Genome Firewall

Predicts which antibiotics will fail against a *Klebsiella
pneumoniae* genome days before laboratory results arrive —
with calibrated confidence, named supporting genes, and an
explicit no-call when the evidence doesn't support an answer.

**Live app:** https://genome-firewall-lln2cdi8zxqy8pnfuinfen.streamlit.app
Hack-Nation 6th Global AI Hackathon · Challenge 06 (OpenAI)

## What it does
FASTA → AMRFinderPlus → gene-presence features → one
calibrated model per drug → verdict, confidence, evidence
category, or no-call. Three drugs: meropenem, ceftazidime,
gentamicin.

## Results (held-out test set, ST258 never seen in training)
| | Meropenem | Ceftazidime | Gentamicin |
|---|---|---|---|
| Balanced accuracy | 0.926 | 0.899 | 0.825 |
| AUROC | 0.944 | 0.947 | 0.875 |
| Brier | 0.081 | 0.060 | 0.115 |
| No-call rate | 4.2% | 4.1% | 28.9% |
| Conformal coverage | 0.897 | 0.901 | 0.891 |

## What didn't hold up
We measured our own safety mechanisms and published the
shortfalls:
- Conformal coverage on gentamicin was 72.4% against a
  nominal 90%. Tightening alpha reached 89.1%, still short,
  at the cost of abstaining on 28.9% of cases.
- The novelty gate raises observed error 6.7%→14.8% on
  meropenem and 13.7%→21.3% on gentamicin, but shows no
  discrimination on ceftazidime. It ships advisory-only.
- Mash validation of the MLST split found a limitation:
  ST437 sits in training at distance 0.0034 from held-out
  ST258, closer than some same-clone pairs.

See the Responsibility tab in the app for all of it.

## Run locally
pip install -r requirements.txt
streamlit run app.py

Two example AMRFinderPlus TSVs are in `examples/`.

## Layout
app.py                Streamlit demo
context/decisions.md  11 architecture decision records
context/feature_spec.md  FASTA→features specification
context/product.md    user, problem, solution, AI role, impact
scripts/              data, features, split, training, gates
models/               calibrated models + conformal thresholds
examples/             sample inputs

## Limitations
One species, three of 74 antibiotics. Gene presence/absence
only. Research prototype — every result requires confirmation
by standard laboratory testing.

## License and data attribution

Code in this repository is MIT licensed (see LICENSE).

This repository does not redistribute source genomic data.
`data/raw/` and `data/interim/` are gitignored; only derived
model artifacts, results, and two example annotation files
are committed.

**Data sources**
- **BV-BRC** (Bacterial and Viral Bioinformatics Resource
  Center, formerly PATRIC) — laboratory-measured antimicrobial
  susceptibility phenotypes and genome annotations. Public
  data, freely available. https://www.bv-brc.org
- **NCBI AMRFinderPlus** (v4.2.7, database 2026-05-15.1) —
  resistance gene annotation. Public domain, unrestricted.
  https://github.com/ncbi/amr
- **Mash** (v2.3) — genome distance estimation for homology
  validation.

BV-BRC specialty-gene annotations draw on several upstream
databases including CARD and NDARO. Their respective terms
govern any reuse of that underlying data. Users obtaining
source data directly should consult those terms.
