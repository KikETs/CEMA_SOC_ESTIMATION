from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime
from .smoothq_retrain import EXPERIMENTS, clone_cfg, configure_strict_training, experiment_cfg, load_relabelled_frames, load_smoothq_lookup
from .thermal_state_model import run_thermal_state_experiment
from .parameter_surface_model import run_parameter_surface_experiment


def _spectral_metrics(x, dt=1.0):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return {
            "spectral_centroid_hz": np.nan,
            "high_frequency_fraction": np.nan,
            "spectral_energy": np.nan,
        }
    x = x - np.mean(x)
    mag = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(len(x), d=float(dt))
    energy = mag ** 2
    total = float(np.sum(energy) + 1e-12)
    centroid = float(np.sum(freq * energy) / total)
    cutoff = np.quantile(freq, 0.75)
    hf = float(np.sum(energy[freq >= cutoff]) / total)
    return {
        "spectral_centroid_hz": centroid,
        "high_frequency_fraction": hf,
        "spectral_energy": total / max(len(x), 1),
    }


def _longest_rest_segment(I, threshold=0.05):
    rest = np.abs(np.asarray(I, dtype=float)) < float(threshold)
    best = cur = 0
    for v in rest:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def compute_excitation_metrics_for_frames(feature_frames, experiment: str) -> pd.DataFrame:
    rows = []
    for split, frames in feature_frames.items():
        for frame in frames:
            f = frame.reset_index(drop=True)
            I = f["I_raw"].to_numpy(np.float64)
            V = f["V_raw"].to_numpy(np.float64)
            dI = np.diff(I, prepend=I[0])
            spec = _spectral_metrics(I)
            current_transition_count = int(np.sum(np.abs(dI) > np.percentile(np.abs(dI), 95))) if len(dI) else 0
            rest_len = _longest_rest_segment(I)
            rest_mask = np.abs(I) < 0.05
            relax_slope = np.nan
            if np.sum(rest_mask) > 10:
                vv = V[rest_mask]
                relax_slope = float(np.nanmedian(np.abs(np.diff(vv)))) if len(vv) > 2 else np.nan
            r0_proxy = float(np.nanvar(V) / (np.nanvar(I) + 1e-9))
            raw_score = (
                np.nanstd(I)
                + 0.15 * np.nanmean(np.abs(dI))
                + 0.02 * current_transition_count
                + 0.001 * rest_len
                + 5.0 * (spec["high_frequency_fraction"] if np.isfinite(spec["high_frequency_fraction"]) else 0.0)
            )
            rows.append({
                "experiment": experiment,
                "split": split,
                "trajectory_id": f["trajectory_id"].iloc[0],
                "drive_cycle": f["drive_cycle"].iloc[0],
                "temperature_C": float(f["temperature"].iloc[0]),
                "n_samples": int(len(f)),
                "current_variance": float(np.nanvar(I)),
                "current_abs_mean": float(np.nanmean(np.abs(I))),
                "dI_abs_energy": float(np.nanmean(np.abs(dI))),
                "current_transition_count": current_transition_count,
                "rest_segment_length_max": rest_len,
                "voltage_relaxation_slope_visibility": relax_slope,
                "R0_identifiability_proxy": r0_proxy,
                "I_spectral_centroid_hz": spec["spectral_centroid_hz"],
                "I_high_frequency_fraction": spec["high_frequency_fraction"],
                "excitation_score": float(raw_score),
            })
    return pd.DataFrame(rows)


def compute_excitation_observability(cfg: CFG | None = None, experiments=("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50")):
    base_cfg = configure_strict_training(clone_cfg(cfg))
    output_dir = base_cfg.output_dir
    lookup = load_smoothq_lookup(output_dir)
    all_rows = []
    for experiment in experiments:
        ecfg = experiment_cfg(base_cfg, experiment)
        configure_strict_training(ecfg)
        frames = load_relabelled_frames(ecfg, experiment, lookup)
        all_rows.append(compute_excitation_metrics_for_frames(frames, experiment))
    metrics = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    metrics.to_csv(output_dir / "excitation_observability_metrics.csv", index=False)
    return metrics


def excitation_vs_error(output_dir: Path, excitation: pd.DataFrame):
    pred_files = [
        output_dir / "thermal_state_prediction_rows.csv",
        output_dir / "parameter_surface_prediction_rows.csv",
        output_dir / "smoothQ_retrain_prediction_rows.csv",
    ]
    preds = []
    for path in pred_files:
        if path.exists():
            df = pd.read_csv(path)
            if "experiment" not in df.columns:
                continue
            preds.append(df)
    if not preds or excitation.empty:
        pd.DataFrame().to_csv(output_dir / "excitation_vs_error.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "excitation_vs_tta_success.csv", index=False)
        return pd.DataFrame()
    pred = pd.concat(preds, ignore_index=True)
    traj_err = pred.groupby(["experiment", "model_name", "trajectory_id"]).agg(
        MAE_pct=("abs_error", lambda x: float(np.mean(x) * 100.0)),
        RMSE_pct=("error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)) * 100.0)),
        voltage_residual_MAE_V=("voltage_residual", lambda x: float(np.mean(np.abs(x))) if np.all(pd.notna(x)) else np.nan),
    ).reset_index()
    merged = traj_err.merge(
        excitation,
        on=["experiment", "trajectory_id"],
        how="left",
        suffixes=("", "_excitation"),
    )
    rows = []
    for (experiment, model), g in merged.groupby(["experiment", "model_name"]):
        for score in ["excitation_score", "current_variance", "dI_abs_energy", "I_high_frequency_fraction"]:
            if score not in g.columns or g[score].isna().all() or len(g) < 3:
                corr = np.nan
            else:
                corr = float(g[score].corr(g["MAE_pct"]))
            rows.append({
                "experiment": experiment,
                "model_name": model,
                "score": score,
                "corr_with_MAE": corr,
            })
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "excitation_vs_error.csv", index=False)
    tta = merged[merged["model_name"].astype(str).str.contains("TTA", na=False)].copy()
    tta.to_csv(output_dir / "excitation_vs_tta_success.csv", index=False)
    lines = [
        "# Excitation / Observability Report",
        "",
        "The excitation score is label-free and uses current variance, current-transition energy, rest-segment visibility, and current DFT metrics.",
        "It is intended as a risk/observability diagnostic, not as a supervised SOC target.",
        "",
    ]
    if len(corr_df):
        lines.extend(["## Error Correlation", corr_df.to_markdown(index=False), ""])
    lines.extend([
        "## Interpretation",
        "- Low excitation should increase SOC uncertainty and should limit aggressive voltage-only adaptation.",
        "- If voltage-only TTA helps only under high excitation, report it as an observability-dependent adaptation result.",
    ])
    (output_dir / "excitation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return corr_df


def _read_optional(path: Path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def aggregate_nextgen_results(output_dir: Path):
    frames = []
    sources = [
        ("smoothQ_baseline", output_dir / "smoothQ_retrain_results.csv", output_dir / "smoothQ_retrain_by_temperature.csv", output_dir / "smoothQ_retrain_omitted_focus.csv"),
        ("thermal_state", output_dir / "thermal_state_results.csv", output_dir / "thermal_state_by_temperature.csv", output_dir / "thermal_state_omitted_focus.csv"),
        ("parameter_surface", output_dir / "parameter_surface_results.csv", output_dir / "parameter_surface_by_temperature.csv", output_dir / "parameter_surface_omitted_focus.csv"),
    ]
    results_all, by_temp_all, focus_all = [], [], []
    for source, res_path, by_path, focus_path in sources:
        res = _read_optional(res_path)
        by = _read_optional(by_path)
        focus = _read_optional(focus_path)
        if len(res):
            res["source"] = source
            results_all.append(res)
        if len(by):
            by["source"] = source
            by_temp_all.append(by)
        if len(focus):
            focus["source"] = source
            focus_all.append(focus)
    results = pd.concat(results_all, ignore_index=True) if results_all else pd.DataFrame()
    by_temp = pd.concat(by_temp_all, ignore_index=True) if by_temp_all else pd.DataFrame()
    focus = pd.concat(focus_all, ignore_index=True) if focus_all else pd.DataFrame()
    if len(results):
        results.to_csv(output_dir / "nextgen_temp_robust_results.csv", index=False)
    if len(by_temp):
        by_temp.to_csv(output_dir / "nextgen_temp_robust_by_temperature.csv", index=False)
    summary_rows = []
    if len(focus):
        omitted = focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])]
        outside = focus[focus["experiment"].isin(["Omit N10", "Omit 50"])]
        for scope, df in [("omitted_A_B_C", omitted), ("outside_range", outside)]:
            for (source, model), g in df.groupby(["source", "model_name"]):
                summary_rows.append({
                    "source": source,
                    "model_name": model,
                    "scope": scope,
                    "average_MAE_pct": float(g["omitted_MAE_pct"].mean()),
                    "worst_MAE_pct": float(g["omitted_MAE_pct"].max()),
                    "average_RMSE_pct": float(g["omitted_RMSE_pct"].mean()),
                    "average_jitter_ratio": float(g["omitted_jitter_ratio"].mean()),
                    "n_folds": int(g["experiment"].nunique()),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "nextgen_temp_robust_focus_summary.csv", index=False)
    return results, by_temp, focus, summary


def write_meta_temperature_proxy(output_dir: Path, focus: pd.DataFrame):
    """Write LOTO-style model-selection tables from completed omitted-temp folds.

    This is deliberately labelled as an evaluation proxy. It does not perform a
    separate episodic inner-loop meta-training run, so reports must not present
    it as true MAML/Reptile-style meta-temperature training.
    """

    if focus.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "meta_temperature_results.csv", index=False)
        empty.to_csv(output_dir / "meta_temperature_loto_results.csv", index=False)
        empty.to_csv(output_dir / "meta_temperature_worst_temp.csv", index=False)
        return empty, empty
    loto = focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])].copy()
    if loto.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "meta_temperature_results.csv", index=False)
        empty.to_csv(output_dir / "meta_temperature_loto_results.csv", index=False)
        empty.to_csv(output_dir / "meta_temperature_worst_temp.csv", index=False)
        return empty, empty
    loto["model_selection_score"] = (
        loto["omitted_MAE_pct"].astype(float)
        + 0.10 * loto["omitted_RMSE_pct"].astype(float)
        + 0.01 * loto["omitted_jitter_ratio"].astype(float)
    )
    loto["meta_training_mode"] = "loto_evaluation_proxy_no_inner_adaptation"
    loto.to_csv(output_dir / "meta_temperature_loto_results.csv", index=False)
    summary = loto.groupby(["source", "model_name"]).agg(
        average_omitted_MAE_pct=("omitted_MAE_pct", "mean"),
        worst_omitted_MAE_pct=("omitted_MAE_pct", "max"),
        average_omitted_RMSE_pct=("omitted_RMSE_pct", "mean"),
        average_jitter_ratio=("omitted_jitter_ratio", "mean"),
        average_model_selection_score=("model_selection_score", "mean"),
        worst_model_selection_score=("model_selection_score", "max"),
        n_loto_folds=("experiment", "nunique"),
    ).reset_index().sort_values("average_model_selection_score")
    summary["meta_training_mode"] = "loto_evaluation_proxy_no_inner_adaptation"
    summary.to_csv(output_dir / "meta_temperature_results.csv", index=False)
    worst = loto.sort_values("omitted_MAE_pct").groupby(["source", "model_name"]).tail(1)
    worst.to_csv(output_dir / "meta_temperature_worst_temp.csv", index=False)
    return summary, loto


def write_nextgen_summary(output_dir: Path, focus_summary: pd.DataFrame):
    lines = [
        "# Next-Generation Temperature Robustness Summary",
        "",
        "This run evaluates internal thermal-state ECM variants, smooth parameter-surface ECM variants, and voltage-only limited TTA diagnostics under the smoothQ physical SOC policy.",
        "",
        "## Main Guardrails",
        "- Train cycles are DST + US06; FUDS remains a supervised test target.",
        "- Voltage-only TTA is adaptation, not pure extrapolation.",
        "- Temperature/core states and parameter surfaces are learned observer states, not directly measured physical quantities.",
        "- Pure unseen-temperature extrapolation is not claimed unless omitted and outside folds improve consistently.",
        "",
    ]
    if len(focus_summary):
        lines.extend(["## Focus Summary", focus_summary.to_markdown(index=False), ""])
        best = focus_summary.sort_values(["scope", "average_MAE_pct"]).groupby("scope").head(3)
        lines.extend(["## Lowest Average MAE By Scope", best.to_markdown(index=False), ""])
    lines.extend([
        "## Safe Claims",
        "- Internal thermal state and parameter surfaces are diagnostic attempts to reduce temperature-domain coverage sensitivity.",
        "- If only TTA improves a target, the result is transductive/online adaptation.",
        "- Excitation metrics can be used as label-free observability risk signals.",
        "",
        "## Forbidden Claims",
        "- The model solves pure unseen-temperature extrapolation.",
        "- TTA results are pure extrapolation.",
        "- Fusion or TTA guarantees robustness at all unseen temperatures.",
        "- Learned voltage components are true physical polarization or hysteresis.",
    ])
    (output_dir / "nextgen_temp_robust_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_nextgen_temp_robust(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50"),
    run_thermal=True,
    run_surface=True,
    include_tta=True,
):
    configure_torch_runtime()
    base_cfg = configure_strict_training(clone_cfg(cfg))
    output_dir = base_cfg.output_dir
    if run_thermal:
        run_thermal_state_experiment(base_cfg, experiments=experiments, include_tta=include_tta)
    if run_surface:
        run_parameter_surface_experiment(base_cfg, experiments=experiments, include_tta=include_tta)
    excitation = compute_excitation_observability(base_cfg, experiments=experiments)
    excitation_vs_error(output_dir, excitation)
    results, by_temp, focus, summary = aggregate_nextgen_results(output_dir)
    write_meta_temperature_proxy(output_dir, focus)
    write_nextgen_summary(output_dir, summary)
    return {
        "results": results,
        "by_temperature": by_temp,
        "focus": focus,
        "summary": summary,
        "excitation": excitation,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run next-generation temperature robustness diagnostics.")
    parser.add_argument("--experiments", default="Exp A,Exp B,Exp C,Omit N10,Omit 50")
    parser.add_argument("--skip-thermal", action="store_true")
    parser.add_argument("--skip-surface", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    exps = tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    out = run_nextgen_temp_robust(
        experiments=exps,
        run_thermal=not args.skip_thermal,
        run_surface=not args.skip_surface,
        include_tta=not args.no_tta,
    )
    print(out["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
