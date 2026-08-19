> **HISTORICAL (superseded)**
>
> This pre-pivot protocol is retained unchanged below as an audit artifact. It is not the current proposed-model protocol. See [`TRANSFER_PROTOCOL_T6_PLAIN.md`](TRANSFER_PROTOCOL_T6_PLAIN.md) for the active T6-plain configuration.

# LFP Confirmatory Transfer Protocol (locked before unblinding)

Protocol lock date: 2026-07-11 KST

## Scope

- Chemistry: LFP A123.
- Label: temperature-matched low-current OCV **discharge** SOC0 and Qref.
- Profiles and rotation: 3-LOPO over DST, FUDS, and US06. Each rotation trains on the other two profiles and tests only the held-out profile.
- Temperatures: `-10, 0, 10, 20, 25, 30, 40, 50 degC`.
- Primary condition: all eight temperatures used for model training and testing.
- Sparse-temperature transfer condition: only `-10, 0, 25, 50 degC` used for model fitting; all eight temperatures used for testing.
- The sparse-temperature condition still estimates temperature-specific R0 at every test temperature from the two training profiles, as required by the locked physical-correction protocol. It does not use the held-out profile.

## Refit and frozen components

Refit independently in every profile rotation:

- `R_hat_0(T)`, using only voltage/current events from that rotation's two LFP training profiles;
- feature-normalization mean and standard deviation, using only that rotation's LFP training profiles and only the model-training temperatures admitted by the condition.

Frozen from the NMC protocol:

- causal EMA sample constants `tau={50,200,800}`;
- window length `L=50` samples and training stride `3` samples;
- GRU residual architecture and all architecture/optimization hyperparameters;
- 200 epochs, final-epoch selection, no validation/test-based checkpoint selection;
- initial seeds `0,1,2`;
- temperature weights `1/1/1` (all LFP temperatures therefore have equal loss weight).

NMC and LFP are both nominally 1 Hz, but the recorded median intervals are not exactly 1.000 s: NMC `1.01549 s` and LFP `1.003004 s`. EMA constants are sample-index constants. Their approximate physical time constants are therefore:

| tau (samples) | NMC | LFP |
|---:|---:|---:|
| 50 | 50.77 s | 50.15 s |
| 200 | 203.10 s | 200.60 s |
| 800 | 812.39 s (13.54 min) | 802.40 s (13.37 min) |

Cold-start plots must use recorded `Test_Time(s)` rather than treating row index as exact seconds.

## Tier 1: feature mini-ablation

Backbone/head is fixed to GRU anchor-residual. No G1, G7, or G8 jobs are admitted.

| Code | Feature set | Channels |
|---|---|---:|
| G0 | `paper_g0_raw` | 3 |
| T6 | `paper_t6_voltage_ema_all` | 9 |
| T7 | `paper_t7_current_abs_ema_all` | 11 |
| G4 | `paper_g4_all_ema` | 17 |

The earlier LFP artifact named G4 used `paper_g4_eqdyn` with 24 channels and is not eligible for reuse. Exact-config completion markers are the only allowed basis for skipping.

## Tier 2: secondary-factor transfer

- Head contrast: GRU-normal with T6 is always run.
- G4-normal is run only if G4 beats T6 outside the locked paired CI in Tier 1.
- Architecture spot-check: window-summary MLP-residual is run only with the Tier-1 selected feature.
- No five-architecture replication is permitted.

## Locked selection rule

For each seed, MAE is first averaged equally over the three held-out profiles and eight test temperatures. Feature differences are paired by seed. The 95% paired t interval is calculated from seeds `0,1,2` before any promotion.

> If T6 LFP performance is within the paired-seed CI of G4, select T6 by parsimony. If T7 or G4 beats T6 outside the CI, report a chemistry-dependent carrier and recommend the superset G4 as the default.

Completeness clause fixed before unblinding: if G0 beats every EMA-bearing candidate outside the paired CI, report failure of EMA transfer and select G0 rather than forcing an EMA carrier conclusion.

After the primary all-eight-temperature Tier-1 decision, only the selected feature is promoted from three to five seeds (`0..4`). Seeds `0..2` are reused and only seeds `3,4` are newly fit. The same selected feature is promoted in both temperature-training conditions so the headline comparison uses the same representation.

After Tier 1, Tier 2, the winner promotion, and the predeclared inference-only Tier 3 diagnostics are complete, no additional LFP experiment is admitted to this suite.

## Tier 3: inference-only diagnostics

No parameter fitting is allowed.

- cold-start error versus recorded elapsed time from existing trajectory starts;
- mid-record forced EMA resets and feature regeneration;
- test-time `R_hat_0` perturbations of `+/-20%` and `+/-50%`;
- `-10 degC` SOC-bin/profile bias decomposition;
- local T6/T7/G4 comparison in voltage-current ambiguity bins;
- OCV-discharge SOC0/Qref label-uncertainty audit, reported separately in SI.

Tier 3 outputs are written below a separate `tier3_inference_only` directory and may not overwrite training artifacts.
