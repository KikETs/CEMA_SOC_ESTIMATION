# Raw-data provenance

## Source

The NMC SP1/SP2 and LFP A123 archives are distributed by the University of
Maryland CALCE Battery Data repository:

- Source page: <https://calce.umd.edu/battery-data>
- Exact archive links and selected workbook names: [README.md](README.md)
- Selected-workbook SHA-256 values: [../reference/raw_source_hashes.csv](../reference/raw_source_hashes.csv)

## Dates

- Local acquisition into this reproducibility workspace: 2026-07-21
- All 28 selected CALCE archive URLs revalidated: 2026-07-22 (28/28 HTTP 200)
- Provenance record updated: 2026-07-23

The 2026-07-21 date is the filesystem creation date shared by the local raw
workbooks when this isolated repository was assembled. It is an acquisition or
copy date, not a claim that every archive was first downloaded from CALCE on
that date. An earlier original-download date cannot be independently recovered
from the preserved files and is therefore not invented here.

## Redistribution boundary

Raw workbooks, downloaded archives, and preprocessed records are local-only and
ignored by Git. The repository tracks source URLs, selected file names,
cryptographic hashes, preprocessing code, and compact protocol metadata. Users
obtain the original workbooks from CALCE and verify them with:

```bash
python scripts/run_reproduction.py preprocess verify-data
```
