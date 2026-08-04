# Methods: leakage-guarded LFP ECM/KF 3-LOPO baseline

## Locked split and reference

The three rotations are `(FUDS,US06)->DST`, `(DST,US06)->FUDS`, and `(DST,FUDS)->US06` at `-10,0,10,20,25,30,40,50 degC`. The SOC reference is the existing low-current OCV-discharge SOC0/Qref label (`SOC_CC`) without relabeling. KF scores use exactly the proposed-model `end_index` mask, starting at index 49. Held-out profile files are excluded by assertion from identification and Q/R selection.

## Current, SOC, and ECM equations

Raw A123 current is negative on discharge. Internally, `I=-Current(A)`, so discharge current is positive. With capacity `Q` in Ah and sample interval `dt` in seconds,

`SOC[k+1] = clip(SOC[k] - I[k] dt/(3600 Q), 0, 1)`.

For branch `j in {1,2}`,

`a_j=exp(-dt/(R_j C_j))`,

`Vp_j[k+1]=a_j Vp_j[k] + (1-a_j) R_j I[k]`.

The terminal-voltage model is

`Vt = OCVbase(SOC,T) + H(SOC,T) h - I R0 - Vp1 - Vp2`.

The charge/discharge low-current OCV branches are independently measured at every test temperature. `OCVbase` is their center and `H>=0` their half-gap. The causal Plett-equivalent one-state hysteresis update is

`alpha_h=exp(-gamma |I| dt/(3600 Q))`,

`h[k+1]=alpha_h h[k] + (1-alpha_h)(-sign(I[k]))`,

with `h in [-1,1]`. Thus discharge tends toward `h=-1` and charge toward `h=+1`.

## Characterization and interpolation

For v2, `R0` is the training-profile-only dV/dI median for each outer fold and each of the eight evaluation temperatures. `R1,R2,tau1,tau2` are least-squares terminal-voltage fits using only that fold's two training profiles at the same temperature. Fit RMSE and `tau1 << tau2` checks are recorded in `results_v2/ecm_fit_quality.csv`.

OCV curves are evaluated in two predeclared forms: raw branch interpolation and a primary monotonic/smoothed form. The primary curve preserves nondecreasing voltage with SOC using accumulated monotonic branch values and PCHIP interpolation. Small plateau slopes remain small; no artificial derivative floor is used. Only extreme numerical derivatives are capped to protect matrix arithmetic.

## Filter safeguards

The plain EKF uses `[SOC,Vp1,Vp2]`; hysteresis EKF/UKF use `[SOC,Vp1,Vp2,h]`. SOC and hysteresis are constrained after every update. EKF covariance uses the Joseph update. Every covariance is symmetrized and eigenvalues are clipped to the predeclared PSD interval. The UKF uses scaled sigma points and the same physical process/measurement models. The adaptive EKF updates measurement variance by bounded innovation EWMA and is reported only when stable.

The primary cold start is `Vp1=Vp2=h=0`. Training-profile-only mean polarization and `h={-1,0,+1}` are sensitivity analyses. Oracle initial SOC uses the first reference SOC only; the non-oracle condition initializes SOC by causal inversion of the first voltage sample with zero polarization/hysteresis. Reference SOC is never used for ECM fitting, OCV fitting, Q/R tuning, or non-oracle initialization.

## Training-only Q/R selection

For each outer holdout, one of the two remaining profiles is fixed as an inner validation profile. `gamma` and Q/R candidates are selected by voltage-innovation RMSE only, without reference SOC. The outer held-out profile is not loaded by the tuning function. Search grids and chosen values are saved verbatim.

## Regions, robustness, and statistics

The OCV-defined regions are locked as plateau `|dOCV/dSOC|<0.10 V/SOC`, transition `0.10-0.50`, and edge `>=0.50`. SOC bands are 0-10, 10-30, 30-70, 70-90, and 90-100%. Sensor biases are illustrative stress tests because no sensor specification is stored with the dataset.

Deterministic filters are not repeated. Any randomized characterization fit would use five fixed seeds. Proposed-versus-KF inference uses fold-temperature paired units and a 10,000-replicate paired bootstrap. Failed/diverged trajectories remain in failure tables and are never silently deleted.

Runtime is measured on this PC. It is not presented as MCU runtime. Scalar-operation counts are model-level engineering estimates and are labeled as such.
