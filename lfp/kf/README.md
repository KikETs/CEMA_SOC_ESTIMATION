# LFP KF paper replay

The supported paper entrypoint is:

```bash
python run_paper_filters.py
```

It evaluates CC, plain/hysteresis/adaptive 2RC-EKF, and hysteresis 2RC-UKF on
the 24 prepared LFP records. The runner uses only:

- the temperature OCV table;
- per-fold, per-temperature R0/RC parameters fitted from the two training
  driving profiles;
- the frozen v2.2 Q/R selections.

Compact paper parameters are under `locked_parameters/`; generated predictions
and metrics are written to the ignored top-level `runs/lfp_kf/` directory.

The runner creates two aggregate views:

- `main_table.csv`: uniform mean over 24 full-evaluation profile-temperature
  slices for every method and initialization;
- `paper_reference_metrics.csv`: the exact aggregation definitions used by the
  supplied paper table. CC oracle, -5 pp, and -10 pp all use the unweighted
  mean of the same 24 full-evaluation profile-temperature slices.

`paper_cc_openloop_slices.csv` retains all CC slice inputs. The
single-thread Zen 3/OpenBLAS and Numba target is configured before NumPy import
by `runtime_env.py`. Stable EKF and CC rows reproduce at `1e-6`; the UKF oracle
aggregate is LAPACK-sensitive on unstable slices and is verified separately at
`0.01 %SOC` without using archived predictions.

`run_all_lopo.py`, `src/parameter_identification.py`, and the v1 audit/report
files are retained only as historical source provenance. They are not called
by the supported paper replay.
