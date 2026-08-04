# Frozen LFP inference package

This directory is the frozen package used for the LFP robustness/deployment
suite. It runs inference without importing the training repository.

- Features: `T6`, `T7`, and `G4`
- Holdouts: `DST`, `FUDS`, and `US06`
- Seeds: `0`, `1`, and `2`
- Temperatures: -10, 0, 10, 20, 25, 30, 40, and 50 C
- Entries: 27 under `{feature}/{fold}/{seed}/`

Each entry contains its original final-epoch weight filename, exact input
configuration, training-profile-only scaler, fold-specific R0 table, frozen
inference code, and a manifest with SHA-256 hashes. The `source_repo` field in
the manifests is historical provenance only; runtime inference does not read
that path.

```python
from lfp.deep_learning.inference_pkg_lfp import load_package

pkg = load_package("lfp/deep_learning/inference_pkg_lfp/G4/US06/0")
soc = pkg.predict(record_dataframe)
```

`golden_test_results.csv` records comparison against the archived prediction
rows. All 27 entries passed at `atol=1e-6`; the largest recorded difference is
below `7e-7`. `validation_summary.json` is the package-level completion gate.
The source evaluation records are needed to repeat golden validation but are
not needed for ordinary inference.

This package deliberately covers seeds 0-2 because those are the models used
by the frozen robustness suite. The five-seed headline result is reproduced by
the training runner and does not imply that all five weights are deployment
packages.
