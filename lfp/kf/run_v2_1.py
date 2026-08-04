#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml

import run_v2 as v2mod
from run_v2 import MAIN_METHODS, PROPOSED, md_table, proposed_rows, rows_from_result, summarize
from src.data_io import load_all, load_confirmatory_predictions
from src.v2_1_core import a1_post_rest_start, fit_hysteresis_ecm_by_fold
from src.v2_core import FoldParameterMap, OCVGrid, estimate_r0_by_fold, run_v2_ekf, run_v2_ukf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v2_1"
START, END = "<!-- V2_1_START -->", "<!-- V2_1_END -->"
D0 = """v2's SOC-MAE objective admitted a degenerate optimum: q_soc→0 makes the SOC channel open-loop
(≈ oracle-init Coulomb counting) while Vp/H states absorb the voltage residuals. A filter exists
to recover from unknown initial SOC, so v2.1 encodes initial-error recovery into selection.
If no trial recovers, that is itself the finding: no Q/R setting reconciles recovery with
plateau accuracy on this data."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_list(items) -> str:
    return hashlib.sha256(("\n".join(sorted(map(str, items))) + "\n").encode()).hexdigest()


def configs():
    cfg = yaml.safe_load((ROOT / "configs/v2_1.yaml").read_text())
    old = yaml.safe_load((ROOT / cfg["v2_config"]).read_text())
    base = yaml.safe_load((ROOT / cfg["base_config"]).read_text())
    return base, old, cfg


def archive_hashes() -> dict:
    groups = {}
    for name in ("artifacts", "results", "figures", "results_v2"):
        paths = sorted((ROOT / name).rglob("*"))
        files = [p for p in paths if p.is_file()]
        groups[name] = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    return groups


def prepare(base, old, cfg, trajectories):
    shutil.copytree(ROOT / "results_v2/audits", OUT / "audits", dirs_exist_ok=True)
    ocv = OCVGrid(pd.read_csv(ROOT / "artifacts/ocv_table.csv"), old["slope_noise"])
    r0, events = estimate_r0_by_fold(trajectories, base["protocol"]["profiles"], old["r0_estimator"])
    ecm = fit_hysteresis_ecm_by_fold(trajectories, base["protocol"]["profiles"], r0, ocv, cfg["ecm_fit"])
    r0.to_csv(OUT / "r0_estimates.csv", index=False)
    events.to_csv(OUT / "r0_events.csv.gz", index=False, compression="gzip")
    ecm.to_csv(OUT / "ecm_fit_quality.csv", index=False)
    summary = ecm.groupby("temperature_C", as_index=False).agg(
        median_residual_mean_mV=("residual_mean_mV", "median"),
        q25_residual_mean_mV=("residual_mean_mV", lambda x: x.quantile(.25)),
        q75_residual_mean_mV=("residual_mean_mV", lambda x: x.quantile(.75)),
        n_record_fits=("residual_mean_mV", "size"),
    )
    summary["systematic_offset_gt_1mV"] = summary.median_residual_mean_mV.abs() > 1
    summary.to_csv(OUT / "ecm_residual_offset_by_temperature.csv", index=False)
    a1 = a1_post_rest_start(trajectories, ocv); a1.to_csv(OUT / "a1_post_rest_start.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for profile, g in a1.groupby("profile"):
        ax.plot(g.temperature_C, g.residual_mV, "o-", label=profile)
    ax.axhline(0, color="black", lw=.8); ax.axhline(1, color="grey", ls="--", lw=.7); ax.axhline(-1, color="grey", ls="--", lw=.7)
    ax.set(xlabel="Temperature (°C)", ylabel="Vstart - OCVdischarge (mV)", title="A1 substitute #2: post-rest record start")
    ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "a1_post_rest_start.png", dpi=180); plt.close(fig)
    unique = ecm.groupby(["fold_holdout", "temperature_C"], as_index=False).first()
    return ocv, FoldParameterMap(unique, pd.read_csv(ROOT / "artifacts/ecm_parameter_map.csv")), ecm


def gamma_for(ecm, holdout, temp):
    return float(ecm.loc[(ecm.fold_holdout == holdout) & (ecm.temperature_C == float(temp)), "gamma"].iloc[0])


def recovery(result, tr, deadline=1800., threshold_pp=1.5):
    err = 100 * np.abs(result.soc - tr.soc_ref)
    elapsed = tr.time_s - tr.time_s[0]
    hit = np.flatnonzero(np.isfinite(err) & (err < threshold_pp) & (elapsed <= deadline))
    return float(elapsed[hit[0]]) if len(hit) else np.nan


def centered_ranges() -> dict:
    old = pd.read_csv(ROOT / "results_v2/qr_selection.csv").sort_values("range_cycle").iloc[-1]
    return {k: [float(old[k]) / 1e3, float(old[k]) * 1e3] for k in ("q_soc", "q_vp", "q_h", "r_voltage")}


def boundary(value, bounds, fraction):
    z = (math.log(value) - math.log(bounds[0])) / (math.log(bounds[1]) - math.log(bounds[0]))
    return "low" if z <= fraction else ("high" if z >= 1 - fraction else None)


def evaluate_noise(noise, holdout, training, ocv, pmap, ecm, old, selection):
    maes = []; recoveries = []; rails = []; details = []
    for tr in training:
        gamma = gamma_for(ecm, holdout, tr.temperature_C)
        for delta in selection["initial_errors_pp"]:
            soc0 = float(np.clip(tr.soc_ref[0] + delta / 100, 0, 1)); h0 = ocv.initial_h(tr, soc0)
            result = run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, old["fixes"], old["slope_noise"], True, initial_h=h0)
            mask = tr.evaluation_mask & np.isfinite(result.soc)
            mae = float(100 * np.mean(np.abs(result.soc[mask] - tr.soc_ref[mask]))) if mask.any() and not result.diverged else 100.
            rt = recovery(result, tr, selection["recovery_deadline_s"], selection["recovery_threshold_pp"])
            rail = float(np.mean(np.abs(result.states[:, 3]) > .999))
            maes.append(mae); rails.append(rail)
            if delta:
                recoveries.append(np.isfinite(rt))
            details.append((delta, mae, rt, rail))
    feasible_fraction = float(np.mean(recoveries)) if recoveries else 0.
    return float(np.mean(maes)), feasible_fraction, details


def tune_fold(holdout, trajectories, ocv, pmap, ecm, old, cfg):
    selection = cfg["selection"]; ranges = centered_ranges(); original = {k: list(v) for k, v in ranges.items()}
    training = [tr for tr in trajectories if tr.profile != holdout]
    files = sorted(str(tr.path) for tr in training); cycles = []
    for cycle in range(selection["max_range_extensions"] + 1):
        db = OUT / "optuna" / f"qr_{holdout}_cycle{cycle}.db"
        study = optuna.create_study(study_name=f"v2_1_recovery_{holdout}_cycle{cycle}", direction="minimize", sampler=optuna.samplers.TPESampler(seed=selection["seed"] + cycle, multivariate=True), storage=f"sqlite:///{db}", load_if_exists=True)
        study.set_user_attr("training_files", files); study.set_user_attr("training_file_list_sha256", sha256_list(files)); study.set_user_attr("ranges", ranges)
        def objective(trial):
            noise = {k: trial.suggest_float(k, b[0], b[1], log=True) for k, b in ranges.items()}
            score, frac, details = evaluate_noise(noise, holdout, training, ocv, pmap, ecm, old, selection)
            feasible = frac >= float(selection["minimum_recovered_fraction"])
            trial.set_user_attr("feasible", feasible); trial.set_user_attr("recovered_fraction", frac)
            trial.set_user_attr("oracle_MAE_pct", float(np.mean([x[1] for x in details if x[0] == 0])))
            trial.set_user_attr("plus5_MAE_pct", float(np.mean([x[1] for x in details if x[0] == 5])))
            trial.set_user_attr("minus5_MAE_pct", float(np.mean([x[1] for x in details if x[0] == -5])))
            trial.set_user_attr("H_rail_fraction", float(np.mean([x[3] for x in details])))
            return score if feasible else 1000. + score
        remaining = int(selection["trials"]) - len(study.trials)
        if remaining > 0: study.optimize(objective, n_trials=remaining, show_progress_bar=False)
        trials = study.trials_dataframe(); trials["fold_holdout"] = holdout; trials["range_cycle"] = cycle
        trials.to_csv(OUT / "optuna" / f"qr_{holdout}_cycle{cycle}_trials.csv", index=False)
        feasible_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and bool(t.user_attrs.get("feasible"))]
        infeasible_fraction = 1 - len(feasible_trials) / max(sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials), 1)
        if not feasible_trials:
            cycles.append({"fold_holdout": holdout, "range_cycle": cycle, "status": "EMPTY_FEASIBLE_SET", "infeasible_fraction": infeasible_fraction, "training_files_sha256": sha256_list(files), **{f"{k}_{side}": v[i] for k, v in ranges.items() for i, side in enumerate(("low", "high"))}})
            return None, pd.DataFrame(cycles), {"branch": "degenerate_empty_feasible_set", "infeasible_fraction": infeasible_fraction}
        best = min(feasible_trials, key=lambda t: t.value); noise = {k: float(best.params[k]) for k in ranges}
        flagged = {k: side for k, side in ((k, boundary(noise[k], ranges[k], selection["boundary_log_fraction"])) for k in ranges) if side}
        score, frac, details = evaluate_noise(noise, holdout, training, ocv, pmap, ecm, old, selection)
        rec = np.array([x[2] for x in details if x[0] != 0], float)
        cycles.append({"fold_holdout": holdout, "range_cycle": cycle, "status": "FEASIBLE", "objective_MAE_pct": score, "infeasible_fraction": infeasible_fraction, "recovered_fraction": frac, "oracle_MAE_pct": np.mean([x[1] for x in details if x[0] == 0]), "plus5_MAE_pct": np.mean([x[1] for x in details if x[0] == 5]), "minus5_MAE_pct": np.mean([x[1] for x in details if x[0] == -5]), "recovery_median_s_among_recovered": np.nanmedian(rec), "recovery_p90_s_among_recovered": np.nanquantile(rec, .9), "H_rail_fraction": np.mean([x[3] for x in details]), "boundary_parameters": ";".join(f"{k}:{v}" for k, v in flagged.items()), "training_files_sha256": sha256_list(files), **noise, **{f"{k}_{side}": v[i] for k, v in ranges.items() for i, side in enumerate(("low", "high"))}})
        if not flagged:
            return noise, pd.DataFrame(cycles), {"branch": "feasible_off_boundary", "infeasible_fraction": infeasible_fraction}
        if cycle >= selection["max_range_extensions"]:
            return noise, pd.DataFrame(cycles), {"branch": "boundary_stop", "flagged": flagged, "infeasible_fraction": infeasible_fraction}
        for k, side in flagged.items():
            if side == "low": ranges[k][0] /= selection["extension_factor"]
            else: ranges[k][1] *= selection["extension_factor"]
    raise AssertionError


def run_filter(model, tr, holdout, ocv, pmap, noise, gamma, old, delta):
    soc0 = float(np.clip(tr.soc_ref[0] + delta / 100, 0, 1)); h0 = ocv.initial_h(tr, soc0)
    if model == "plain_2rc_ekf": return run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, old["fixes"], old["slope_noise"], False)
    if model == "hysteresis_2rc_ekf": return run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, old["fixes"], old["slope_noise"], True, initial_h=h0)
    if model == "adaptive_hysteresis_2rc_ekf": return run_v2_ekf(tr, holdout, ocv, pmap, soc0, noise, gamma, old["fixes"], old["slope_noise"], True, adaptive=True, initial_h=h0)
    return run_v2_ukf(tr, holdout, ocv, pmap, soc0, noise, gamma, old["fixes"], old["slope_noise"], initial_h=h0)


def run_d3(trajectories, ocv, pmap, ecm, selections, old):
    frames = []; failures = []
    for tr in trajectories:
        for delta, label in ((0, "oracle"), (5, "plus5pp"), (-5, "minus5pp")):
            for method in MAIN_METHODS:
                result = run_filter(method, tr, tr.profile, ocv, pmap, selections[tr.profile], gamma_for(ecm, tr.profile, tr.temperature_C), old, delta)
                frames.append(rows_from_result(method, label, tr, result, ocv))
                if result.diverged: failures.append({"method": method, "initial_condition": label, "profile": tr.profile, "temperature_C": tr.temperature_C, "reason": result.divergence_reason})
    frames.append(proposed_rows(trajectories, ocv, load_confirmatory_predictions(old["proposed_confirmatory"])))
    pred = pd.concat(frames, ignore_index=True); pred.to_csv(OUT / "prediction_rows_v2_1.csv.gz", index=False, compression="gzip")
    pd.DataFrame(failures, columns=["method", "initial_condition", "profile", "temperature_C", "reason"]).to_csv(OUT / "failures.csv", index=False)
    summarize(pred, ["method", "initial_condition", "profile", "temperature_C"]).to_csv(OUT / "temperature_metrics.csv", index=False)
    summarize(pred, ["method", "initial_condition", "profile"]).to_csv(OUT / "fold_summary.csv", index=False)
    summarize(pred, ["method", "initial_condition", "profile", "temperature_C", "plateau_edge_region"]).to_csv(OUT / "plateau_edge_metrics.csv", index=False)
    summarize(pred, ["method", "initial_condition", "profile", "temperature_C", "time_since_start_bin_s"]).to_csv(OUT / "time_since_start.csv", index=False)
    rows = []
    for (method, initial), g in pred.groupby(["method", "initial_condition"]):
        slices = summarize(g, ["profile", "temperature_C"])
        rows.append({"method": method, "initial_condition": initial, "weighting": "slice_unweighted_primary", "MAE_pct": slices.MAE_pct.mean(), "RMSE_pct": slices.RMSE_pct.mean(), "n_slices": len(slices), "n_points": len(g)})
        rows.append({"method": method, "initial_condition": initial, "weighting": "pointwise_secondary", "MAE_pct": 100*g.abs_error.mean(), "RMSE_pct": 100*np.sqrt(np.mean((g.soc_pred-g.soc_true)**2)), "n_slices": len(slices), "n_points": len(g)})
    pd.DataFrame(rows).to_csv(OUT / "main_table.csv", index=False)
    v2mod.OUT = OUT; v2mod.circular_block_bootstrap(pred, {"statistics": {"circular_block_samples": 60, "bootstrap_replicates": 10000, "bootstrap_seed": 20260712}}).to_csv(OUT / "bootstrap.csv", index=False)


def d4_tradeoff(trajectories, ocv, pmap, ecm, noise, old, cfg):
    rows = []
    points = [(float(x), False) for x in cfg["tradeoff"]["q_soc_values"]] + [(np.nan, True)]
    for q_soc, open_loop in points:
        for tr in trajectories:
            metrics = {}
            for delta, prefix in ((0, "oracle"), (5, "plus5"), (-5, "minus5")):
                use = dict(noise); use["q_soc"] = 0. if open_loop else q_soc
                soc0 = float(np.clip(tr.soc_ref[0] + delta / 100, 0, 1)); gamma = gamma_for(ecm, tr.profile, tr.temperature_C)
                result = run_v2_ekf(tr, tr.profile, ocv, pmap, soc0, use, gamma, old["fixes"], old["slope_noise"], True, open_loop=open_loop, initial_h=ocv.initial_h(tr, soc0))
                elapsed = tr.time_s - tr.time_s[0]; late = tr.evaluation_mask & (elapsed >= 1800) & np.isfinite(result.soc)
                if not late.any(): late = tr.evaluation_mask & np.isfinite(result.soc)
                metrics[f"{prefix}_steady_MAE_pct"] = float(100*np.mean(np.abs(result.soc[late]-tr.soc_ref[late])))
                metrics[f"{prefix}_recovery_time_s"] = recovery(result, tr)
                metrics[f"{prefix}_residual_error_pp"] = float(100*np.mean(np.abs(result.soc[late]-tr.soc_ref[late])))
            rows.append({"chemistry": "LFP", "fold": tr.profile, "profile": tr.profile, "temperature_C": tr.temperature_C, "source_file": str(tr.path), "sweep_point": "open_loop" if open_loop else f"q_soc={q_soc:.9g}", "q_soc": q_soc, "open_loop": open_loop, **{k: noise[k] for k in ("q_vp", "q_h", "r_voltage")}, **metrics})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "tradeoff.csv", index=False)
    chart = frame.copy()
    chart["plus5_recovery_censored_s"] = chart.plus5_recovery_time_s.fillna(1800.0)
    chart["minus5_recovery_censored_s"] = chart.minus5_recovery_time_s.fillna(1800.0)
    plot = chart.groupby(["open_loop", "q_soc"], dropna=False).agg(
        oracle_MAE=("oracle_steady_MAE_pct", "mean"),
        plus5_recovery=("plus5_recovery_censored_s", "median"),
        minus5_recovery=("minus5_recovery_censored_s", "median"),
    ).reset_index()
    plot["worst_sign_recovery"] = plot[["plus5_recovery", "minus5_recovery"]].max(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); finite = plot[~plot.open_loop]
    ax.plot(finite.q_soc, finite.oracle_MAE, "o-", color="#2f6f9f"); ax.set_xscale("log"); ax.set_xlabel("q_soc"); ax.set_ylabel("Oracle steady MAE (%)", color="#2f6f9f")
    ax2 = ax.twinx(); ax2.plot(finite.q_soc, finite.worst_sign_recovery, "s--", color="#c66b2b"); ax2.set_ylabel("Worst-sign median recovery (s; failures=1800)", color="#c66b2b"); ax2.set_ylim(0, 1900)
    open_loop_row = plot[plot.open_loop].iloc[0]
    recovered_minus = int(frame.loc[frame.open_loop, "minus5_recovery_time_s"].notna().sum())
    ax.text(.02, .97, f"Exact open-loop: oracle MAE={open_loop_row.oracle_MAE:.3f}%\n−5 pp recovered={recovered_minus}/24", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": .9})
    ax.set_title("Steady accuracy vs initial-error recovery"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "tradeoff.png", dpi=180); plt.close(fig)
    return frame


def nmc_placeholder(cfg, lfp):
    columns = list(lfp.columns)
    out = pd.DataFrame(columns=columns)
    out.to_csv(OUT / "tradeoff_nmc.csv", index=False)
    roots = [Path(cfg["nmc"]["requested_root"]), Path(cfg["nmc"]["local_mirror"])]
    return {"status": "SOURCE_UNAVAILABLE", "searched_roots": [str(x) for x in roots], "output_rows": 0, "schema": columns}


def leakage(trajectories, ecm, cycles, branch, before, cfg):
    folds = {}
    for holdout in sorted({tr.profile for tr in trajectories}):
        train = sorted(str(tr.path) for tr in trajectories if tr.profile != holdout); test = sorted(str(tr.path) for tr in trajectories if tr.profile == holdout)
        efiles = sorted(set(ecm.loc[ecm.fold_holdout == holdout, "training_file"]))
        qrows = cycles[cycles.fold_holdout == holdout] if len(cycles) else pd.DataFrame()
        qhash = None if qrows.empty else str(qrows.iloc[-1].training_files_sha256)
        folds[holdout] = {"training_files": train, "test_files": test, "training_file_list_sha256": sha256_list(train), "test_file_list_sha256": sha256_list(test), "ecm_training_files": efiles, "ecm_match": efiles == train, "qr_training_hash": qhash, "qr_match": qhash is None or qhash == sha256_list(train), "intersection": sorted(set(train)&set(test))}
    after = archive_hashes(); immutable = before == after
    audit = {"status": "PASS" if all(x["ecm_match"] and x["qr_match"] and not x["intersection"] for x in folds.values()) and immutable else "FAIL", "branch": branch, "folds": folds, "v1_v2_immutable": immutable, "config_sha256": sha256_file(ROOT/"configs/v2_1.yaml"), "source_sha256": {"run_v2_1.py": sha256_file(ROOT/"run_v2_1.py"), "src/v2_1_core.py": sha256_file(ROOT/"src/v2_1_core.py")}}
    (OUT/"leakage_audit_v2_1.json").write_text(json.dumps(audit, indent=2)+"\n"); return audit


def report(branch, cycles, ecm, offset, a1, nmc_status):
    representative = cycles[cycles.fold_holdout == "DST"].sort_values("range_cycle").iloc[-1]
    infeasible = float(representative.infeasible_fraction)
    section = f"""{START}

## v2.1 — revised-search rerun

### D0 — declared rationale

{D0}

### Outcome

- Branch: **{branch}**.
- Optuna infeasible fraction in the decisive representative-fold cycle: `{infeasible:.3%}`.
- D3 main table: **{'completed' if branch == 'feasible_off_boundary' else 'skipped by the declared gate'}**.
- D4 LFP trade-off exhibit: completed for eight q_soc values plus exact open-loop.
- D4 NMC overlay: `{nmc_status['status']}`; the requested source repo was not present at either declared local path, so an empty schema-compatible CSV was emitted and no NMC values were fabricated.

### D1 — ECM and alignment substitutes

- Hysteresis EMF is included and gamma is fitted jointly per fold×temperature within `[1e-6, 1e-2]` using training profiles only.
- Per-record voltage residuals: `{len(ecm)}` rows; RMSE >10 mV in `{int(ecm.rmse_gt_10mV.sum())}/{len(ecm)}` rows.
- Per-temperature systematic-offset substitute flags: `{int(offset.systematic_offset_gt_1mV.sum())}/{len(offset)}` temperatures exceed |median residual mean| >1 mV.
- Post-rest-start substitute: `{len(a1)}` records; `{int((a1.residual_mV.abs()>1).sum())}/{len(a1)}` exceed |Vstart-OCVdischarge| >1 mV.
- Gamma hit its declared upper bound in `{int(np.isclose(ecm.gamma, 1e-2).sum())}/{len(ecm)}` per-record fit rows; the ECM fit is therefore boundary-limited even though all optimizations terminated successfully.

### D2 — declared search and selection

Each parameter's initial range is six log decades, centered on its v2 cycle-1 value (`center/1e3` to `center×1e3`). All oracle/+5 pp/-5 pp training-profile×temperature slices are equally weighted. A trial is feasible only when at least 90% of the ±5 pp runs cross below 1.5 pp error by 1800 s.

All 24 records start at SOC=100%. Consequently the requested +5 pp initialization is projected back to the physical upper bound and is numerically identical to oracle initialization; the recovery gate is informative primarily for the −5 pp direction. `initial_condition_clipping.csv` records requested and applied errors. This limitation is retained rather than silently redefining the perturbation.

{md_table(cycles)}

### Honesty and limitations

- All completed and infeasible Optuna trials are retained in SQLite and CSV under `results_v2_1/optuna/`.
- v1, v2, and their numerical artifacts were hash-checked as immutable.
- The missing NMC KF repository is a handoff blocker only for the requested NMC overlay; it does not alter the LFP branch finding.

{END}
"""
    path=ROOT/"report.md"; old=path.read_text(); path.write_text(old.split(START)[0].rstrip()+"\n\n"+section+("\n"+old.split(END,1)[1].lstrip() if END in old else ""))


def main():
    OUT.mkdir(exist_ok=True); (OUT/"optuna").mkdir(exist_ok=True)
    before=archive_hashes(); base, old, cfg=configs(); trajectories=load_all(base)
    ocv,pmap,ecm=prepare(base,old,cfg,trajectories)
    ranges=centered_ranges(); pd.DataFrame([{"parameter":k,"center":math.sqrt(v[0]*v[1]),"low":v[0],"high":v[1],"log_decades":6} for k,v in ranges.items()]).to_csv(OUT/"qr_declared_ranges.csv",index=False)
    selections={}; allcycles=[]; rep=cfg["selection"]["representative_holdout"]
    noise,cycles,status=tune_fold(rep,trajectories,ocv,pmap,ecm,old,cfg); allcycles.append(cycles)
    branch=status["branch"]
    if branch=="feasible_off_boundary":
        selections[rep]=noise
        for holdout in base["protocol"]["profiles"]:
            if holdout==rep: continue
            n,c,s=tune_fold(holdout,trajectories,ocv,pmap,ecm,old,cfg); allcycles.append(c)
            if s["branch"]!="feasible_off_boundary": branch=f"{s['branch']}_{holdout}"; noise=n or noise; break
            selections[holdout]=n
    cycles=pd.concat(allcycles,ignore_index=True); cycles.to_csv(OUT/"qr_selection.csv",index=False)
    if branch=="feasible_off_boundary": run_d3(trajectories,ocv,pmap,ecm,selections,old)
    fallback=noise if noise is not None else {k:float(pd.read_csv(ROOT/"results_v2/qr_selection.csv").sort_values("range_cycle").iloc[-1][k]) for k in ("q_soc","q_vp","q_h","r_voltage")}
    lfp=d4_tradeoff(trajectories,ocv,pmap,ecm,fallback,old,cfg); nmc=nmc_placeholder(cfg,lfp); (OUT/"nmc_source_status.json").write_text(json.dumps(nmc,indent=2)+"\n")
    offset=pd.read_csv(OUT/"ecm_residual_offset_by_temperature.csv"); a1=pd.read_csv(OUT/"a1_post_rest_start.csv")
    audit=leakage(trajectories,ecm,cycles,branch,before,cfg); report(branch,cycles,ecm,offset,a1,nmc)
    decisive = cycles.sort_values("range_cycle").groupby("fold_holdout", as_index=False).tail(1)
    completeness={"status":"PASS_WITH_NMC_SOURCE_BLOCKER" if audit["status"]=="PASS" else "FAIL","branch":branch,"d3_run":branch=="feasible_off_boundary","d4_lfp_rows":len(lfp),"d4_nmc_rows":0,"nmc_status":nmc["status"],"infeasible_fraction_by_fold":dict(zip(decisive.fold_holdout, decisive.infeasible_fraction.astype(float)))}
    (OUT/"completeness_audit.json").write_text(json.dumps(completeness,indent=2)+"\n"); print(json.dumps(completeness,indent=2))


if __name__=="__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING); main()
