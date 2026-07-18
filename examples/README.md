# Example AMRFinderPlus TSVs

Upload either file on the app's **Report** tab (Source → "Upload AMRFinderPlus TSV")
to exercise the upload path. Each is the AMRFinderPlus `--nucleotide` output for one
BV-BRC *Klebsiella pneumoniae* genome. Known lab phenotypes (meropenem / ceftazidime / gentamicin):

- `example_resistant_KPC_573.24328.tsv` — BV-BRC accession **573.24328** — Resistant / Resistant / Resistant (carries `blaKPC-3`; expect meropenem evidence category (i)).
- `example_susceptible_573.14328.tsv` — BV-BRC accession **573.14328** — Susceptible / Susceptible / Susceptible (no carbapenemase/ESBL; expect no known-determinant hit).
