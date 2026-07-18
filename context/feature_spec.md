# Feature spec: FASTA -> AMR gene-presence features

Documents the path from a downloaded genome assembly to a genotypic
feature vector, as exercised on the stratified 20-genome validation
subset (`data/interim/subset20.txt`: 10 meropenem-Resistant, 10
meropenem-Susceptible, seed=42, drawn from
`data/interim/cohort_labels.available.csv` — see
`scripts/select_subset20.py`).

## 1. Input: assembled nucleotide FASTA

- Source: BV-BRC (PATRIC) FTP, one file per genome id.
  ```
  curl --ssl-reqd --user anonymous:guest -f -s \
    -o data/interim/fasta/{genome_id}.fna \
    ftp://ftp.bv-brc.org/genomes/{genome_id}/{genome_id}.fna
  ```
- Format: standard multi-contig nucleotide FASTA. Headers follow BV-BRC's
  convention, e.g.
  `>573.13136.con.0001   NODE_1_length_505278_cov_43.2098   [Klebsiella pneumoniae KPN1135ec | 573.13136]`
  — draft assemblies, tens to ~100+ contigs per genome, ~5.4-5.9 MB total
  per genome in this subset.
- No reference genome or annotation is required upstream; AMRFinderPlus
  does its own ORF calling from raw nucleotide sequence in `--nucleotide`
  mode.

## 2. AMRFinderPlus command

```bash
amrfinder --nucleotide data/interim/fasta/{genome_id}.fna \
  --organism Klebsiella_pneumoniae --plus --threads 8 \
  -o data/interim/amrfinder/{genome_id}.tsv
```

- `--organism Klebsiella_pneumoniae` enables the organism-specific point-
  mutation database (e.g. `gyrA`/`ompK36` QRDR mutations) on top of the
  core AMR gene database; without it only the organism-agnostic acquired-
  gene search runs.
- `--plus` additionally reports stress-response and virulence genes
  (`Type` = `STRESS` / `VIRULENCE`) alongside `AMR` calls.
- Runtime on this hardware (conda env `amrfinder`, AMRFinderPlus 4.2.7,
  database 2026-05-15.1): ~11-29s per genome, ~340s wall time for all 20
  genomes sequentially with `--threads 8`. See
  `data/interim/amrfinder_timing.log`.

## 3. Output columns

Tab-separated, one row per gene/mutation call:

| Column | Meaning |
|---|---|
| Protein id | Protein accession if run in protein mode; `NA` for `--nucleotide` mode |
| Contig id | Assembly contig the hit is on |
| Start / Stop / Strand | Coordinates of the hit on the contig |
| Element symbol | Gene/allele symbol, e.g. `blaKPC-2`, `gyrA_D87N` |
| Element name | Human-readable name |
| Scope | `core` (curated gene) or `plus` (plus-database gene) |
| Type | `AMR`, `STRESS`, or `VIRULENCE` |
| Subtype | `AMR` (acquired gene), `POINT` (point mutation), `POINT_DISRUPT`, or `METAL`/etc. under STRESS |
| Class / Subclass | Drug class / subclass conferred, e.g. `BETA-LACTAM` / `CARBAPENEM` |
| Method | Call method: `ALLELEX` (exact allele), `BLASTX`/`BLASTP`, `POINTX` (point mutation), `PARTIAL`, `HMM`, etc. |
| Target length / Reference sequence length / % Coverage / % Identity / Alignment length | Match quality metrics |
| Closest reference accession / name | Nearest curated reference protein |
| HMM accession / description | HMM hit info, when the call is HMM-based |

## 4. Presence/absence encoding rule

For a fixed feature vocabulary (e.g. a curated list of clinically
relevant beta-lactamase/carbapenemase families: `KPC`, `NDM`, `OXA`,
`CTX-M`, `SHV`, `TEM`, plus any other genes of interest), encode each
genome as a binary feature vector:

```
feature[gene_family] = 1  if any row in the genome's AMRFinderPlus
                           output has Subtype == "AMR" (acquired-gene
                           call, not a point mutation) AND its
                           `Element symbol`, after normalization,
                           starts with that gene_family token
                        else 0
```

- Normalization (`scripts/validate_amrfinder.py:normalize`): lowercase,
  strip a leading `bla` prefix, strip non-alphanumeric characters, strip
  trailing allele-number digits. E.g. `blaKPC-2` -> `kpc`, `blaSHV-28` ->
  `shv`, `blaCTX-M-15` -> `ctxm`.
- Restricting to `Subtype == "AMR"` excludes point-mutation rows
  (`POINT`/`POINT_DISRUPT`), which represent chromosomal variants rather
  than acquired genes and are not comparable to simple gene
  presence/absence.
- `Scope`/`Type` are not filtered further here — `--plus` STRESS/
  VIRULENCE rows are excluded by the `Type == "AMR"` implication of
  `Subtype == "AMR"` (spot-checked: no non-AMR-type row carries
  `Subtype == "AMR"` in this dataset).
- Multiple alleles of the same family in one genome (e.g. two distinct
  `blaSHV-*` calls) collapse to a single `1` for that family — this is a
  presence/absence encoding, not an allele-count or allele-identity
  encoding.

## 5. Validation against spgene.tab (this subset)

`scripts/validate_amrfinder.py` compares AMRFinderPlus's `Subtype ==
"AMR"` gene calls against the `property == "Antibiotic Resistance"` rows
in each genome's `data/interim/spgene/{genome_id}.spgene.tab` (BV-BRC's
independently-computed PATRIC/CARD/VFDB/ARDB annotation), after applying
the same family-level normalization to both sides (see ADR-010 in
`context/decisions.md` for why gene-name comparison requires this, and
why raw agreement is nonetheless low). Full output:
`data/interim/validation_report.txt`.

Headline result for the presence/absence encoding above: **10/10**
meropenem-resistant genomes in this subset carry an AMRFinderPlus
acquired-gene call in at least one of KPC/NDM/OXA/CTX-M/SHV/TEM
(`blaKPC-2`/`blaKPC-3` in 9/10, `blaNDM-1` in 1/10, plus `blaSHV-*`/
`blaOXA`/`blaCTX-M-15`/`blaTEM*` variously) — consistent with these being
real carbapenem-resistance determinants and the encoding correctly
surfacing them.
