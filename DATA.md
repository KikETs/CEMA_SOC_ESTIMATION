# Data and reproduction contract

Only user-supplied raw workbooks belong under `Data/`. Raw workbooks and every
generated trajectory are ignored by Git.

The exact official archives, selected workbook names, checksums, and access
dates are documented in [Data/README.md](Data/README.md) and
[Data/PROVENANCE.md](Data/PROVENANCE.md).

```text
Data/
  NMC/
    OCV/       # 3 low-current OCV workbooks: 0, 25, 45 C
    Profiles/  # 50SOC and 80SOC workbooks may coexist; paper run selects 80SOC
  LFP/
    OCV/       # A1-007 OCV workbooks at -10, 0, 10, 20, 25, 30, 40, 50 C
    Profiles/  # A1-007 DST/US06/FUDS workbooks, N10 denotes -10 C
  Preprocessed/  # generated; never commit
```

The reported LFP v2.2 KF baseline uses the per-fold, per-temperature ECM fit
obtained from the two training profiles. Its compact frozen ECM/Q-R protocol
tables are under `lfp/kf/locked_parameters/`.

Run and verify both preprocessing chains:

```bash
python scripts/run_reproduction.py preprocess verify-data
```

Every stage writes stdout/stderr logs below `Data/logs/`. A failed chemistry
does not suppress the other chemistry, and all failures appear in
`Data/Preprocessed/preprocessing_status.csv`. The runner finishes the other
chemistry, then exits nonzero if any stage is `FAIL` or was `SKIP`ped because an
upstream record failed.
