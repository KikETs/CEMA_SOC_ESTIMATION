from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .training import attach_prediction_features, build_prediction_feature_lookup
from .variance_control import variance_by_temperature, _overall_metrics
from .extrapolation_robustness import load_filtered_feature_frames
from .extrapolation_robustness import train_rex_model, attach_and_summarize
from .neural_ecm_observer import (
    ECMSpec,
    adapt_voltage_only,
    ecm_exp_cfg,
    omitted_temp_for,
    predict_full_trajectories,
    train_ecm_model,
)


FOCUS_EXPERIMENTS = ("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50")
PREFIX_EXPERIMENTS = ("Exp C", "Omit N10", "Omit 50")
PREFIX_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.20, 1.0)
BEST_R5_AUG_REX_BY_EXP = {
    "Exp A": "R5_GATED_AUG_REX_l2p0",
    "Exp B": "R5_GATED_AUG_REX_l0p1",
    "Exp C": "R5_GATED_AUG_REX_l1p0",
}
ECM_FUSION_MODELS = ("NeuralECMObserver_REX", "NeuralECMObserver_REX_TTA_voltage_only")


def _repo_root(cfg: CFG | None = None) -> Path:
    return Path((cfg or make_cfg()).output_dir)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _vraw_column(df: pd.DataFrame) -> str:
    for col in ("V_raw", "V_raw_x", "V_raw_y"):
        if col in df.columns:
            return col
    raise KeyError("No V_raw column found")


def _ensure_phase(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trajectory_fraction" not in out.columns:
        out["trajectory_fraction"] = np.nan
        for _, idx in out.groupby(["experiment", "model_name", "trajectory_id"]).groups.items():
            ii = np.asarray(list(idx))
            n = len(ii)
            out.loc[ii, "trajectory_fraction"] = np.linspace(0.0, 1.0, n)
    out["phase_bin"] = pd.cut(
        out["trajectory_fraction"].astype(float).clip(0, 1),
        bins=[-1e-9, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["early", "mid", "late"],
        include_lowest=True,
    )
    return out


def _metrics(g: pd.DataFrame) -> dict:
    y_true = g["y_true"].to_numpy(np.float64)
    y_pred = g["y_pred"].to_numpy(np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)
    row = {
        "n_samples": int(len(g)),
        "MAE": float(np.mean(abs_err)) if len(g) else np.nan,
        "MAE_pct": float(np.mean(abs_err) * 100.0) if len(g) else np.nan,
        "RMSE": float(np.sqrt(np.mean(err**2))) if len(g) else np.nan,
        "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0) if len(g) else np.nan,
        "error_std": float(np.std(err)) if len(g) else np.nan,
        "error_std_pct": float(np.std(err) * 100.0) if len(g) else np.nan,
        "max_error_pct": float(np.max(abs_err) * 100.0) if len(g) else np.nan,
    }
    if len(g) > 2:
        order = np.argsort(g["end_index"].to_numpy(np.float64))
        yp = y_pred[order]
        yt = y_true[order]
        dp = np.diff(yp)
        dt = np.diff(yt)
        pred_jitter = np.mean(np.abs(dp))
        true_jitter = np.mean(np.abs(dt))
        row.update(
            {
                "pred_jitter": float(pred_jitter),
                "true_jitter": float(true_jitter),
                "jitter_ratio": float(pred_jitter / max(true_jitter, 1e-8)),
                "delta_soc_mae": float(np.mean(np.abs(dp - dt))),
                "high_frequency_error_energy": float(np.mean(np.diff(err[order]) ** 2)),
            }
        )
    else:
        row.update(
            {
                "pred_jitter": np.nan,
                "true_jitter": np.nan,
                "jitter_ratio": np.nan,
                "delta_soc_mae": np.nan,
                "high_frequency_error_energy": np.nan,
            }
        )
    return row


def _safe_corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return np.nan
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _target_rows(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    omitted = omitted_temp_for(experiment)
    return df[(df["experiment"].eq(experiment)) & np.isclose(df["temperature_C"].astype(float), omitted)].copy()


def parameter_curve_diagnostics(root: Path) -> pd.DataFrame:
    params = _read_csv(root / "neural_ecm_parameter_curves.csv")
    focus = _read_csv(root / "neural_ecm_omitted_temp_focus.csv")
    vres = _read_csv(root / "neural_ecm_voltage_residual.csv")

    fig, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    axes = axes.ravel()
    plot_params = ["Q_eff", "R0", "tau1", "tau2", "hys_gain", "hys_tau", "OCV"]
    mid_soc = params.iloc[(params["SOC"] - 0.5).abs().argsort()[:1]]["SOC"].iloc[0]
    mid = params[np.isclose(params["SOC"], mid_soc)].copy()
    for ax, col in zip(axes, plot_params):
        for (exp, model), g in mid.groupby(["experiment", "model_name"], sort=True):
            g = g.sort_values("temperature_C")
            style = "-" if model.endswith("_REX") else "--"
            alpha = 0.85 if model.endswith("_REX") else 0.45
            ax.plot(g["temperature_C"], g[col], style, alpha=alpha, linewidth=1.5, label=f"{exp} {model}")
        ax.set_title(f"{col} at SOC~{mid_soc:.2f}")
        ax.set_xlabel("temperature_C")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center", fontsize=7, ncol=1)
    fig.savefig(root / "neural_ecm_parameter_curves_by_fold.png", dpi=180)
    plt.close(fig)

    rows = []
    for (exp, model), g in params.groupby(["experiment", "model_name"]):
        r0_viol = 0
        r0_steps = 0
        for _, sg in g.groupby("SOC"):
            sg = sg.sort_values("temperature_C")
            diff = np.diff(sg["R0"].to_numpy(np.float64))
            r0_viol += int(np.sum(diff > 0))
            r0_steps += int(len(diff))
        row = {
            "experiment": exp,
            "model_name": model,
            "R0_monotonic_violation_fraction": float(r0_viol / max(r0_steps, 1)),
        }
        for col in ["Q_eff", "R0", "tau1", "tau2", "hys_gain", "hys_tau", "OCV"]:
            vals = g[col].to_numpy(np.float64)
            row[f"{col}_min"] = float(np.nanmin(vals))
            row[f"{col}_max"] = float(np.nanmax(vals))
            row[f"{col}_range"] = float(np.nanmax(vals) - np.nanmin(vals))
        rows.append(row)
    plaus = pd.DataFrame(rows)

    report = []
    report.append("# NeuralECM Parameter Plausibility Report\n")
    report.append("This diagnostic uses the strict window_len=50, stride=1, max 300 epoch plateau-stopped NeuralECM run.\n")
    report.append("The curves are learned observer parameters, not independently identified electrochemical parameters.\n")
    for exp in FOCUS_EXPERIMENTS:
        e_focus = focus[focus["experiment"].eq(exp)].copy()
        e_vres = vres[vres["experiment"].eq(exp)].copy()
        if e_focus.empty:
            continue
        omitted = omitted_temp_for(exp)
        report.append(f"\n## {exp} omitted/target {omitted:g}C\n")
        for model in ["NeuralECMObserver", "NeuralECMObserver_REX", "NeuralECMObserver_REX_TTA_voltage_only"]:
            row = e_focus[e_focus["model_name"].eq(model)]
            vr = e_vres[e_vres["model_name"].eq(model)]
            if row.empty:
                continue
            bits = [
                f"- {model}: omitted MAE {row['omitted_MAE_pct'].iloc[0]:.2f}%, "
                f"RMSE {row['omitted_RMSE_pct'].iloc[0]:.2f}%, "
                f"jitter ratio {row['omitted_jitter_ratio'].iloc[0]:.2f}"
            ]
            if len(vr):
                bits.append(f", voltage residual MAE {vr['voltage_MAE_V'].iloc[0] * 1000:.2f} mV")
            report.append("".join(bits) + "\n")
        if exp == "Exp C":
            report.append(
                "Interpretation: the REX observer is strong at omitted 20C; low SOC error and low voltage residual move together here.\n"
            )
        elif exp in ("Exp A", "Exp B"):
            report.append(
                "Interpretation: voltage residual remains only a few mV while SOC error is high, so voltage fit alone is not enough; OCV/Q_eff/observer-state mapping is likely miscalibrated for this omitted transition temperature.\n"
            )
        elif exp == "Omit N10":
            report.append(
                "Interpretation: voltage-only TTA gives a practical low-temperature adaptation signal, but it is transductive adaptation, not pure extrapolation.\n"
            )
        elif exp == "Omit 50":
            report.append(
                "Interpretation: both SOC error and voltage residual remain high, so the high-temperature outside-range dynamic/parameter extrapolation is still weak.\n"
            )

    report.append("\n## Plausibility Checks\n")
    for _, row in plaus.iterrows():
        report.append(
            f"- {row['experiment']} {row['model_name']}: R0 monotonic violation fraction "
            f"{row['R0_monotonic_violation_fraction']:.3f}, Q_eff range "
            f"{row['Q_eff_min']:.3f}-{row['Q_eff_max']:.3f}, R0 range "
            f"{row['R0_min']:.4f}-{row['R0_max']:.4f}.\n"
        )
    report.append("\nSafe statement: NeuralECM observer constraints reduce jitter and improve some omitted/outside folds, but they do not solve pure temperature extrapolation.\n")
    report.append("Forbidden statement: NeuralECM solves unseen-temperature extrapolation.\n")
    (root / "neural_ecm_parameter_plausibility_report.md").write_text("".join(report), encoding="utf-8")
    return plaus


def voltage_vs_soc_error_diagnostics(root: Path) -> pd.DataFrame:
    pred = _ensure_phase(_read_csv(root / "neural_ecm_prediction_rows.csv"))
    pred["abs_voltage_residual"] = pred["voltage_residual"].abs()
    rows = []
    group_cols = ["experiment", "model_name"]
    for keys, g in pred.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "scope": "overall",
                "temperature_C": np.nan,
                "SOC_bin": "all",
                "phase_bin": "all",
                "voltage_residual_mae": float(g["abs_voltage_residual"].mean()),
                "voltage_residual_rmse": float(np.sqrt(np.mean(g["voltage_residual"].to_numpy(np.float64) ** 2))),
                "SOC_mae": float(g["abs_error"].mean()),
                "SOC_rmse": float(np.sqrt(np.mean(g["error"].to_numpy(np.float64) ** 2))),
                "corr_abs_voltage_residual_abs_SOC_error": _safe_corr(g["abs_voltage_residual"], g["abs_error"]),
                "n_samples": int(len(g)),
            }
        )
        rows.append(row)
    for keys, g in pred.groupby(["experiment", "model_name", "temperature_C"]):
        row = dict(zip(["experiment", "model_name", "temperature_C"], keys))
        row.update(
            {
                "scope": "temperature",
                "SOC_bin": "all",
                "phase_bin": "all",
                "voltage_residual_mae": float(g["abs_voltage_residual"].mean()),
                "voltage_residual_rmse": float(np.sqrt(np.mean(g["voltage_residual"].to_numpy(np.float64) ** 2))),
                "SOC_mae": float(g["abs_error"].mean()),
                "SOC_rmse": float(np.sqrt(np.mean(g["error"].to_numpy(np.float64) ** 2))),
                "corr_abs_voltage_residual_abs_SOC_error": _safe_corr(g["abs_voltage_residual"], g["abs_error"]),
                "n_samples": int(len(g)),
            }
        )
        rows.append(row)
    for keys, g in pred.groupby(["experiment", "model_name", "SOC_bin"]):
        row = dict(zip(["experiment", "model_name", "SOC_bin"], keys))
        row.update(
            {
                "scope": "soc_bin",
                "temperature_C": np.nan,
                "phase_bin": "all",
                "voltage_residual_mae": float(g["abs_voltage_residual"].mean()),
                "voltage_residual_rmse": float(np.sqrt(np.mean(g["voltage_residual"].to_numpy(np.float64) ** 2))),
                "SOC_mae": float(g["abs_error"].mean()),
                "SOC_rmse": float(np.sqrt(np.mean(g["error"].to_numpy(np.float64) ** 2))),
                "corr_abs_voltage_residual_abs_SOC_error": _safe_corr(g["abs_voltage_residual"], g["abs_error"]),
                "n_samples": int(len(g)),
            }
        )
        rows.append(row)
    for keys, g in pred.groupby(["experiment", "model_name", "phase_bin"], observed=False):
        if len(g) == 0:
            continue
        row = dict(zip(["experiment", "model_name", "phase_bin"], keys))
        row.update(
            {
                "scope": "phase",
                "temperature_C": np.nan,
                "SOC_bin": "all",
                "voltage_residual_mae": float(g["abs_voltage_residual"].mean()),
                "voltage_residual_rmse": float(np.sqrt(np.mean(g["voltage_residual"].to_numpy(np.float64) ** 2))),
                "SOC_mae": float(g["abs_error"].mean()),
                "SOC_rmse": float(np.sqrt(np.mean(g["error"].to_numpy(np.float64) ** 2))),
                "corr_abs_voltage_residual_abs_SOC_error": _safe_corr(g["abs_voltage_residual"], g["abs_error"]),
                "n_samples": int(len(g)),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(root / "neural_ecm_voltage_vs_soc_error.csv", index=False)

    plot_dir = root / "neural_ecm_voltage_residual_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for exp in FOCUS_EXPERIMENTS:
        target = _target_rows(pred, exp)
        if target.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, model in zip(axes, ["NeuralECMObserver", "NeuralECMObserver_REX", "NeuralECMObserver_REX_TTA_voltage_only"]):
            g = target[target["model_name"].eq(model)]
            if g.empty:
                ax.axis("off")
                continue
            ax.scatter(g["abs_voltage_residual"] * 1000.0, g["abs_error"] * 100.0, s=3, alpha=0.25)
            ax.set_title(f"{exp} {model}")
            ax.set_xlabel("|voltage residual| mV")
            ax.set_ylabel("|SOC error| %")
            ax.grid(True, alpha=0.25)
        fig.savefig(plot_dir / f"{exp.replace(' ', '_')}_voltage_residual_vs_soc_error.png", dpi=170)
        plt.close(fig)
    return out


def _predict_with_prefix_scales(model, frames, model_name: str, scales_by_tid: dict[str, torch.Tensor]) -> pd.DataFrame:
    return predict_full_trajectories(model, frames, model_name, scales_by_tid=scales_by_tid)


def adapt_voltage_only_prefix(
    model,
    frame: pd.DataFrame,
    cfg: CFG,
    prefix_fraction: float,
    *,
    steps: int = 12,
    lr: float = 0.03,
    drift: float = 0.02,
) -> torch.Tensor:
    if prefix_fraction <= 0:
        return torch.zeros(5, device=device)
    if prefix_fraction >= 0.999:
        return adapt_voltage_only(model, frame, cfg, steps=steps, lr=lr, drift=drift)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    scales = torch.nn.Parameter(torch.zeros(5, device=device))
    opt = torch.optim.Adam([scales], lr=float(lr))
    f = frame.reset_index(drop=True)
    n_prefix = max(8, int(np.ceil(len(f) * float(prefix_fraction))))
    prefix = f.iloc[:n_prefix].reset_index(drop=True)
    I_np = prefix["I_raw"].to_numpy(np.float32)
    V_np = prefix["V_raw"].to_numpy(np.float32)
    T_np = prefix["temperature"].to_numpy(np.float32)
    n = len(prefix)
    chunk = min(int(getattr(cfg, "ecm_tta_chunk_len", 512)), n)
    rng = np.random.default_rng(123)
    for _ in range(int(steps)):
        starts = [0] if n <= chunk else rng.integers(0, n - chunk + 1, size=min(4, max(1, n - chunk + 1))).tolist()
        loss = scales.new_tensor(0.0)
        for st in starts:
            sl = slice(st, st + chunk)
            I = torch.as_tensor(I_np[sl][None, :], device=device)
            V = torch.as_tensor(V_np[sl][None, :], device=device)
            T = torch.as_tensor(T_np[sl][None, :], device=device)
            out = model.rollout(I, V, T, soc0=torch.ones(1, device=device), scales=scales)
            loss = loss + F.smooth_l1_loss(out["voltage"], V, beta=0.03)
        loss = loss / len(starts) + float(drift) * torch.mean(scales**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            scales.clamp_(-np.log(2.0), np.log(2.0))
    for p in model.parameters():
        p.requires_grad_(True)
    return scales.detach()


def run_tta_prefix_experiment(cfg: CFG | None = None, experiments=PREFIX_EXPERIMENTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_torch_runtime()
    cfg = cfg or make_cfg()
    all_pred = []
    scale_rows = []
    for experiment in experiments:
        ecfg = ecm_exp_cfg(cfg, experiment)
        feature_frames = load_filtered_feature_frames(ecfg, decomposed_dir=ecfg.decomposed_dir)
        omitted = omitted_temp_for(experiment)
        spec = ECMSpec(
            name="NeuralECMObserver_REX",
            use_rex=True,
            lambda_v=0.20,
            lambda_rex=0.5,
            lambda_worst=0.10,
            correction_limit=0.05,
        )
        model, history, _ = train_ecm_model(feature_frames, ecfg, spec, experiment)
        history.to_csv(ecfg.output_dir / f"neural_ecm_tta_prefix_training_history_{experiment.replace(' ', '_')}.csv", index=False)
        target_frames = [
            f for f in feature_frames["test"] if np.isclose(float(f["temperature"].iloc[0]), float(omitted))
        ]
        for frac in PREFIX_FRACTIONS:
            scales = {}
            prefix_name = "no_tta" if frac <= 0 else ("full_trajectory_tta" if frac >= 0.999 else f"prefix_{int(frac * 100)}pct")
            for frame in target_frames:
                tid = str(frame["trajectory_id"].iloc[0])
                s = adapt_voltage_only_prefix(
                    model,
                    frame,
                    ecfg,
                    frac,
                    steps=int(getattr(ecfg, "ecm_tta_steps", 12)),
                )
                scales[tid] = s
                scale_rows.append(
                    {
                        "experiment": experiment,
                        "target_temperature_C": float(omitted),
                        "prefix_fraction": float(frac),
                        "prefix_name": prefix_name,
                        "trajectory_id": tid,
                        "log_Q_scale": float(s[0].cpu()),
                        "log_R0_scale": float(s[1].cpu()),
                        "log_RC_gain_scale": float(s[2].cpu()),
                        "log_tau_scale": float(s[3].cpu()),
                        "log_hys_gain_scale": float(s[4].cpu()),
                    }
                )
            pred = _predict_with_prefix_scales(
                model,
                target_frames,
                f"NeuralECMObserver_REX_TTA_{prefix_name}",
                scales,
            )
            if len(pred):
                pred["experiment"] = experiment
                pred["prefix_fraction"] = float(frac)
                pred["prefix_name"] = prefix_name
                all_pred.append(pred)
    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    scales_all = pd.DataFrame(scale_rows)
    return pred_all, scales_all


def summarize_tta_prefix(root: Path, pred: pd.DataFrame, scales: pd.DataFrame) -> None:
    if pred.empty:
        pd.DataFrame().to_csv(root / "neural_ecm_tta_prefix_results.csv", index=False)
        pd.DataFrame().to_csv(root / "neural_ecm_tta_prefix_by_target.csv", index=False)
        return
    rows = []
    for keys, g in pred.groupby(["experiment", "prefix_fraction", "prefix_name"]):
        row = dict(zip(["experiment", "prefix_fraction", "prefix_name"], keys))
        row["target_temperature_C"] = float(g["temperature_C"].iloc[0])
        row.update(_metrics(g))
        vres = g["voltage_residual"].to_numpy(np.float64)
        row["voltage_residual_mae"] = float(np.mean(np.abs(vres)))
        row["voltage_residual_rmse"] = float(np.sqrt(np.mean(vres**2)))
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["experiment", "prefix_fraction"])
    res.to_csv(root / "neural_ecm_tta_prefix_results.csv", index=False)
    res.to_csv(root / "neural_ecm_tta_prefix_by_target.csv", index=False)
    scales.to_csv(root / "neural_ecm_tta_prefix_scales.csv", index=False)

    plot_dir = root / "neural_ecm_tta_prefix_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for exp, g in res.groupby("experiment"):
        fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax1.plot(g["prefix_fraction"] * 100, g["MAE_pct"], marker="o", label="SOC MAE %")
        ax1.plot(g["prefix_fraction"] * 100, g["RMSE_pct"], marker="s", label="SOC RMSE %")
        ax1.set_xlabel("target prefix used for voltage-only TTA (%)")
        ax1.set_ylabel("SOC error (%)")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(g["prefix_fraction"] * 100, g["voltage_residual_mae"] * 1000, color="tab:green", marker="^", label="voltage MAE mV")
        ax2.set_ylabel("voltage residual MAE (mV)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.savefig(plot_dir / f"{exp.replace(' ', '_')}_tta_prefix_curve.png", dpi=170)
        plt.close(fig)


def _prepare_r5_for_fusion(root: Path) -> pd.DataFrame:
    r5 = _read_csv(root / "rex_prediction_rows.csv")
    keep = []
    for exp, name in BEST_R5_AUG_REX_BY_EXP.items():
        g = r5[r5["experiment"].eq(exp) & r5["model_name"].eq(name)].copy()
        if len(g):
            keep.append(g)
    if not keep:
        return pd.DataFrame()
    out = pd.concat(keep, ignore_index=True)
    out = out.rename(columns={"y_pred": "SOC_R5", "abs_error": "abs_error_R5", "error": "error_R5"})
    return out


def run_outside_r5_prediction_artifacts(
    cfg: CFG | None = None,
    *,
    experiments=("Omit N10", "Omit 50"),
    lambda_rex: float = 1.0,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    configure_torch_runtime()
    cfg = cfg or make_cfg()
    root = _repo_root(cfg)
    pred_path = root / "outside_r5_prediction_rows.csv"
    if pred_path.exists() and not force:
        pred = pd.read_csv(pred_path)
        have = set(pred.get("experiment", pd.Series(dtype=str)).dropna().unique())
        if set(experiments).issubset(have):
            return {
                "predictions": pred,
                "results": pd.read_csv(root / "outside_r5_results.csv") if (root / "outside_r5_results.csv").exists() else pd.DataFrame(),
                "by_temperature": pd.read_csv(root / "outside_r5_by_temperature.csv") if (root / "outside_r5_by_temperature.csv").exists() else pd.DataFrame(),
                "focus": pd.read_csv(root / "outside_r5_focus.csv") if (root / "outside_r5_focus.csv").exists() else pd.DataFrame(),
            }
    all_pred, all_results, all_by_temp, all_focus, all_hist, all_loss = [], [], [], [], [], []
    for experiment in experiments:
        ecfg = ecm_exp_cfg(cfg, experiment)
        feature_frames = load_filtered_feature_frames(ecfg, decomposed_dir=ecfg.decomposed_dir)
        omitted = omitted_temp_for(experiment)
        model_name = f"R5_GATED_AUG_REX_l{str(lambda_rex).replace('.', 'p')}_outside"
        _, hist, loss_by_temp, pred = train_rex_model(
            feature_frames,
            ecfg,
            model_name,
            lambda_rex=float(lambda_rex),
            use_aug=True,
            experiment=experiment,
        )
        attached, overall, by_temp, focus = attach_and_summarize(
            [(model_name, pred)],
            feature_frames,
            ecfg,
            experiment,
            omitted,
        )
        all_pred.append(attached)
        all_results.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
        all_hist.append(hist)
        all_loss.append(loss_by_temp)
    out = {
        "predictions": pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame(),
        "results": pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(),
        "by_temperature": pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame(),
        "focus": pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame(),
        "history": pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame(),
        "loss_by_temperature": pd.concat(all_loss, ignore_index=True) if all_loss else pd.DataFrame(),
    }
    out["predictions"].to_csv(root / "outside_r5_prediction_rows.csv", index=False)
    out["results"].to_csv(root / "outside_r5_results.csv", index=False)
    out["by_temperature"].to_csv(root / "outside_r5_by_temperature.csv", index=False)
    out["focus"].to_csv(root / "outside_r5_focus.csv", index=False)
    out["history"].to_csv(root / "outside_r5_training_history.csv", index=False)
    out["loss_by_temperature"].to_csv(root / "outside_r5_loss_by_temperature.csv", index=False)
    return out


def _prepare_ecm_for_fusion(root: Path) -> pd.DataFrame:
    ecm = _read_csv(root / "neural_ecm_prediction_rows.csv")
    keep_models = ["NeuralECMObserver_REX", "NeuralECMObserver_REX_TTA_voltage_only"]
    return ecm[ecm["model_name"].isin(keep_models)].copy()


def _robust_z(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 1e-12 else vals.std()
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return ((vals - med) / scale).clip(-6, 6)


def _causal_rolling_features(
    df: pd.DataFrame,
    pred_col: str,
    *,
    residual_col: str | None = None,
    window: int = 50,
    prefix: str = "",
) -> pd.DataFrame:
    out = df.sort_values(["experiment", "model_name", "trajectory_id", "end_index"]).copy()
    pred_diff = out.groupby(["experiment", "model_name", "trajectory_id"])[pred_col].diff().abs().fillna(0.0)
    out[f"{prefix}recent_jitter"] = (
        pred_diff.groupby([out["experiment"], out["model_name"], out["trajectory_id"]])
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
        .to_numpy()
    )
    if residual_col is not None and residual_col in out.columns:
        abs_resid = out[residual_col].abs()
        sq_resid = out[residual_col].astype(float) ** 2
        grp = [out["experiment"], out["model_name"], out["trajectory_id"]]
        out[f"{prefix}residual_abs_mean"] = (
            abs_resid.groupby(grp).rolling(window, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).to_numpy()
        )
        out[f"{prefix}residual_rms"] = np.sqrt(
            sq_resid.groupby(grp).rolling(window, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).to_numpy()
        )
    return out


def _fixed_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64).clip(-30, 30)))


def _causal_smooth_weight(df: pd.DataFrame, raw_col: str, out_col: str, alpha: float = 0.18) -> pd.DataFrame:
    out = df.sort_values(["experiment", "fusion_model", "trajectory_id", "end_index"]).copy()
    vals = np.zeros(len(out), dtype=np.float64)
    for _, idx in out.groupby(["experiment", "fusion_model", "trajectory_id"], sort=False).groups.items():
        locs = np.asarray(list(idx))
        raw = out.loc[locs, raw_col].to_numpy(np.float64)
        sm = np.empty_like(raw)
        prev = raw[0] if len(raw) else 0.5
        for i, v in enumerate(raw):
            prev = float(alpha * v + (1.0 - alpha) * prev) if i else float(v)
            sm[i] = prev
        vals[out.index.get_indexer(locs)] = sm
    out[out_col] = vals.clip(0.02, 0.98)
    return out


def _load_r5_auxiliary_risk(root: Path) -> pd.DataFrame:
    pieces = []
    ood_path = root / "ood_feature_distance_scores.csv"
    if ood_path.exists():
        cols = [
            "trajectory_id",
            "end_index",
            "distance_S4_latent",
            "mahalanobis_S3",
            "knn_S4_k5",
            "gate_entropy",
            "gate_variation",
        ]
        ood = pd.read_csv(ood_path, usecols=lambda c: c in cols)
        if len(ood):
            ood = ood.groupby(["trajectory_id", "end_index"], as_index=False).mean(numeric_only=True)
            pieces.append(ood)
    unc_path = root / "ood_prediction_uncertainty.csv"
    if unc_path.exists():
        cols = [
            "model_name",
            "trajectory_id",
            "end_index",
            "ensemble_std",
            "prediction_range",
            "gate_entropy",
            "gate_variation",
            "distance_S4_latent",
            "mahalanobis_S3",
            "knn_S4_k5",
        ]
        unc = pd.read_csv(unc_path, usecols=lambda c: c in cols)
        if len(unc):
            preferred = unc[unc["model_name"].str.contains("R5_GATED_AUG", na=False)].copy()
            if preferred.empty:
                preferred = unc[unc["model_name"].eq("R5_GATED")].copy()
            preferred = preferred.groupby(["trajectory_id", "end_index"], as_index=False).mean(numeric_only=True)
            pieces.append(preferred)
    if not pieces:
        return pd.DataFrame(columns=["trajectory_id", "end_index"])
    aux = pieces[0]
    for p in pieces[1:]:
        aux = aux.merge(p, on=["trajectory_id", "end_index"], how="outer", suffixes=("", "_unc"))
    for base in ["distance_S4_latent", "mahalanobis_S3", "knn_S4_k5", "gate_entropy", "gate_variation"]:
        alt = f"{base}_unc"
        if alt in aux.columns:
            if base in aux.columns:
                aux[base] = aux[base].combine_first(aux[alt])
            else:
                aux[base] = aux[alt]
            aux = aux.drop(columns=[alt])
    return aux


def _load_tta_parameter_drift(root: Path) -> pd.DataFrame:
    path = root / "neural_ecm_tta_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["experiment", "trajectory_id", "ecm_parameter_drift"])
    tta = pd.read_csv(path)
    if tta.empty:
        return pd.DataFrame(columns=["experiment", "trajectory_id", "ecm_parameter_drift"])
    scale_cols = [c for c in tta.columns if c.startswith("log_") and c.endswith("_scale")]
    if not scale_cols:
        return pd.DataFrame(columns=["experiment", "trajectory_id", "ecm_parameter_drift"])
    tta["ecm_parameter_drift"] = np.sqrt(np.sum(np.square(tta[scale_cols].to_numpy(np.float64)), axis=1))
    return tta[["experiment", "trajectory_id", "ecm_parameter_drift"]].copy()


def dual_estimator_fusion(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r5 = _prepare_r5_for_fusion(root)
    ecm = _prepare_ecm_for_fusion(root)
    if r5.empty or ecm.empty:
        empty = pd.DataFrame()
        empty.to_csv(root / "dual_estimator_fusion_results.csv", index=False)
        empty.to_csv(root / "dual_estimator_fusion_by_temperature.csv", index=False)
        empty.to_csv(root / "dual_estimator_fusion_omitted_focus.csv", index=False)
        return empty, empty, empty

    join_cols = ["experiment", "trajectory_id", "end_index"]
    r5_cols = join_cols + ["SOC_R5", "y_true", "temperature_C", "drive_cycle"]
    if "trajectory_fraction" in r5.columns:
        r5_cols.append("trajectory_fraction")
    r5_small = r5[r5_cols].copy()
    rows = []
    for ecm_model, eg in ecm.groupby("model_name"):
        merged = eg.merge(r5_small, on=join_cols, suffixes=("_ECM", "_R5"), how="inner")
        if merged.empty:
            continue
        if "temperature_C_ECM" in merged.columns:
            merged["temperature_C"] = merged["temperature_C_ECM"]
        if "drive_cycle_ECM" in merged.columns:
            merged["drive_cycle"] = merged["drive_cycle_ECM"]
        merged["SOC_ECM"] = merged["y_pred"]
        merged["SOC_true"] = merged["y_true_ECM"] if "y_true_ECM" in merged.columns else merged["y_true"]
        merged["ecm_abs_voltage_residual"] = merged["voltage_residual"].abs()
        merged["prediction_disagreement"] = (merged["SOC_ECM"] - merged["SOC_R5"]).abs()
        merged["risk_ecm"] = merged.groupby("experiment")["ecm_abs_voltage_residual"].transform(_robust_z)
        merged["risk_disagreement"] = merged.groupby("experiment")["prediction_disagreement"].transform(_robust_z)
        # Label-free fixed rule: high ECM voltage residual or high disagreement shifts
        # weight back toward R5. No SOC labels are used to set this gate.
        merged["w_ecm"] = 1.0 / (1.0 + np.exp(merged["risk_ecm"] + 0.35 * merged["risk_disagreement"]))
        merged["w_ecm"] = merged["w_ecm"].clip(0.05, 0.95)
        merged["SOC_fused"] = merged["w_ecm"] * merged["SOC_ECM"] + (1.0 - merged["w_ecm"]) * merged["SOC_R5"]
        merged["y_pred"] = merged["SOC_fused"]
        merged["y_true"] = merged["SOC_true"]
        merged["error"] = merged["y_pred"] - merged["y_true"]
        merged["abs_error"] = merged["error"].abs()
        merged["model_name"] = f"Fusion_label_free_{ecm_model}"
        rows.append(merged)
    fused = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    result_rows = []
    by_temp_rows = []
    focus_rows = []
    for (exp, model), g in fused.groupby(["experiment", "model_name"]):
        row = {"experiment": exp, "model_name": model}
        row.update(_metrics(g))
        result_rows.append(row)
        for temp, tg in g.groupby("temperature_C"):
            tr = {"experiment": exp, "model_name": model, "temperature_C": float(temp)}
            tr.update(_metrics(tg))
            tr["mean_w_ecm"] = float(tg["w_ecm"].mean())
            by_temp_rows.append(tr)
        omitted = omitted_temp_for(exp)
        og = g[np.isclose(g["temperature_C"].astype(float), omitted)]
        if len(og):
            fr = {"experiment": exp, "model_name": model, "omitted_temperature_C": float(omitted)}
            fr.update({f"omitted_{k}": v for k, v in _metrics(og).items()})
            fr["mean_w_ecm_omitted"] = float(og["w_ecm"].mean())
            focus_rows.append(fr)
    results = pd.DataFrame(result_rows)
    by_temp = pd.DataFrame(by_temp_rows)
    focus = pd.DataFrame(focus_rows)
    results.to_csv(root / "dual_estimator_fusion_results.csv", index=False)
    by_temp.to_csv(root / "dual_estimator_fusion_by_temperature.csv", index=False)
    focus.to_csv(root / "dual_estimator_fusion_omitted_focus.csv", index=False)
    fused.to_csv(root / "dual_estimator_fusion_prediction_rows.csv", index=False)

    if len(by_temp):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for model, g in by_temp.groupby("model_name"):
            gg = g.groupby("temperature_C", as_index=False)["mean_w_ecm"].mean()
            ax.plot(gg["temperature_C"], gg["mean_w_ecm"], marker="o", label=model)
        ax.set_xlabel("temperature_C")
        ax.set_ylabel("mean ECM fusion weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.savefig(root / "fusion_weight_by_temperature.png", dpi=170)
        plt.close(fig)

    baseline_focus = _read_csv(root / "neural_ecm_omitted_temp_focus.csv")
    report = ["# Fusion Risk Report\n\n"]
    report.append("Fusion uses a fixed label-free gate from ECM voltage residual and R5/ECM prediction disagreement. Test SOC labels are not used for fusion decisions.\n\n")
    for exp in ("Exp A", "Exp B", "Exp C"):
        report.append(f"## {exp}\n")
        b = baseline_focus[baseline_focus["experiment"].eq(exp)]
        for name in ["R5_GATED", BEST_R5_AUG_REX_BY_EXP.get(exp), "NeuralECMObserver_REX", "NeuralECMObserver_REX_TTA_voltage_only"]:
            if not name:
                continue
            row = b[b["model_name"].eq(name)]
            if len(row):
                report.append(f"- {name}: omitted MAE {row['omitted_MAE_pct'].iloc[0]:.2f}%, jitter {row['omitted_jitter_ratio'].iloc[0]:.2f}\n")
        f = focus[focus["experiment"].eq(exp)]
        for _, row in f.iterrows():
            report.append(
                f"- {row['model_name']}: omitted MAE {row['omitted_MAE_pct']:.2f}%, "
                f"jitter {row['omitted_jitter_ratio']:.2f}, mean ECM weight {row['mean_w_ecm_omitted']:.2f}\n"
            )
        report.append("\n")
    report.append("Interpretation: if fusion does not improve worst omitted folds, the simple label-free residual/disagreement gate is insufficient; that is a negative result, not evidence of solved extrapolation.\n")
    (root / "fusion_risk_report.md").write_text("".join(report), encoding="utf-8")
    return results, by_temp, focus


def _prepare_jitter_fusion_rows(root: Path) -> pd.DataFrame:
    r5 = _prepare_r5_for_fusion(root)
    ecm = _prepare_ecm_for_fusion(root)
    if r5.empty or ecm.empty:
        return pd.DataFrame()
    r5 = r5.rename(columns={"model_name": "r5_model_name"}).copy()
    r5["model_name"] = r5["r5_model_name"]
    r5 = _causal_rolling_features(r5, "SOC_R5", window=50, prefix="r5_")
    r5 = r5.drop(columns=["model_name"])
    r5_aux = _load_r5_auxiliary_risk(root)
    if not r5_aux.empty:
        r5 = r5.merge(r5_aux, on=["trajectory_id", "end_index"], how="left")

    ecm = _causal_rolling_features(ecm, "y_pred", residual_col="voltage_residual", window=50, prefix="ecm_")
    drift = _load_tta_parameter_drift(root)
    if not drift.empty:
        ecm = ecm.merge(drift, on=["experiment", "trajectory_id"], how="left")
    ecm["ecm_parameter_drift"] = ecm["ecm_parameter_drift"].fillna(0.0) if "ecm_parameter_drift" in ecm else 0.0

    join_cols = ["experiment", "trajectory_id", "end_index"]
    r5_cols = [
        "experiment",
        "trajectory_id",
        "end_index",
        "SOC_R5",
        "y_true",
        "temperature_C",
        "drive_cycle",
        "trajectory_fraction",
        "r5_model_name",
        "r5_recent_jitter",
    ]
    for col in [
        "distance_S4_latent",
        "mahalanobis_S3",
        "knn_S4_k5",
        "ensemble_std",
        "prediction_range",
        "gate_entropy",
        "gate_variation",
    ]:
        if col in r5.columns:
            r5_cols.append(col)
    r5_small = r5[[c for c in r5_cols if c in r5.columns]].copy()

    rows = []
    for ecm_model, eg in ecm[ecm["model_name"].isin(ECM_FUSION_MODELS)].groupby("model_name"):
        merged = eg.merge(r5_small, on=join_cols, suffixes=("_ECM", "_R5"), how="inner")
        if merged.empty:
            continue
        if "temperature_C_ECM" in merged:
            merged["temperature_C"] = merged["temperature_C_ECM"]
        elif "temperature_C" not in merged and "temperature_C_R5" in merged:
            merged["temperature_C"] = merged["temperature_C_R5"]
        if "drive_cycle_ECM" in merged:
            merged["drive_cycle"] = merged["drive_cycle_ECM"]
        elif "drive_cycle" not in merged and "drive_cycle_R5" in merged:
            merged["drive_cycle"] = merged["drive_cycle_R5"]
        if "trajectory_fraction_ECM" in merged:
            merged["trajectory_fraction"] = merged["trajectory_fraction_ECM"]
        elif "trajectory_fraction" not in merged and "trajectory_fraction_R5" in merged:
            merged["trajectory_fraction"] = merged["trajectory_fraction_R5"]
        merged["SOC_ECM"] = merged["y_pred"]
        merged["SOC_true"] = merged["y_true_ECM"] if "y_true_ECM" in merged else merged["y_true"]
        merged["ecm_model_name"] = ecm_model
        merged["prediction_disagreement"] = (merged["SOC_ECM"] - merged["SOC_R5"]).abs()
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    defaults = {
        "distance_S4_latent": 8.0,
        "mahalanobis_S3": 3.0,
        "knn_S4_k5": 3.0,
        "ensemble_std": 0.0,
        "prediction_range": 0.0,
        "gate_entropy": 0.0,
        "gate_variation": 0.0,
        "r5_recent_jitter": 0.0,
        "ecm_recent_jitter": 0.0,
        "ecm_residual_abs_mean": 0.0,
        "ecm_residual_rms": 0.0,
        "ecm_parameter_drift": 0.0,
    }
    for col, val in defaults.items():
        if col not in out:
            out[col] = val
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(val)
    return out


def _add_label_free_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Fixed, label-free scales. These are not fit on test labels and use only
    # causal/local prediction or voltage residual quantities.
    out["ecm_residual_risk"] = (out["ecm_residual_rms"] / 0.012).clip(0, 8)
    out["ecm_drift_risk"] = (out["ecm_parameter_drift"] / 0.35).clip(0, 8)
    out["ecm_jitter_risk"] = (out["ecm_recent_jitter"] / 0.0015).clip(0, 8)
    out["r5_jitter_risk"] = (out["r5_recent_jitter"] / 0.006).clip(0, 8)
    out["r5_ood_risk"] = (
        0.35 * (out["distance_S4_latent"] / 12.0).clip(0, 8)
        + 0.35 * (out["mahalanobis_S3"] / 5.0).clip(0, 8)
        + 0.30 * (out["knn_S4_k5"] / 4.0).clip(0, 8)
    )
    out["r5_uncertainty_risk"] = (
        0.60 * (out["ensemble_std"] / 0.004).clip(0, 8)
        + 0.40 * (out["prediction_range"] / 0.012).clip(0, 8)
    )
    out["disagreement_risk"] = (out["prediction_disagreement"] / 0.05).clip(0, 8)
    out["ecm_risk"] = (
        0.95 * out["ecm_residual_risk"]
        + 0.55 * out["ecm_drift_risk"]
        + 0.40 * out["ecm_jitter_risk"]
    )
    out["r5_risk"] = (
        0.80 * out["r5_jitter_risk"]
        + 0.55 * out["r5_ood_risk"]
        + 0.45 * out["r5_uncertainty_risk"]
        + 0.15 * out["gate_entropy"].fillna(0.0)
        + 0.15 * (out["gate_variation"].fillna(0.0) / 0.05).clip(0, 8)
    )
    return out


def _make_fusion_variant(base: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = _add_label_free_risk_scores(base)
    if variant == "jitter_aware_rule":
        raw = _fixed_sigmoid(1.35 * (out["r5_risk"] - out["ecm_risk"]) - 0.20 * out["disagreement_risk"])
        low_resid = out["ecm_residual_rms"] < 0.008
        high_r5_jitter = out["r5_recent_jitter"] > 0.004
        high_ecm_risk = (out["ecm_residual_rms"] > 0.020) | (out["ecm_parameter_drift"] > 0.55)
        raw = np.where(low_resid & high_r5_jitter, np.maximum(raw, 0.78), raw)
        raw = np.where(high_ecm_risk, np.minimum(raw, 0.22), raw)
    elif variant == "inverse_risk":
        raw = 1.0 / (1.0 + np.exp((out["ecm_risk"] - out["r5_risk"]).clip(-30, 30)))
    elif variant == "jitter_aware_mlp_proxy":
        # Label-free proxy for an MLP-style nonlinear gate. A supervised MLP is
        # intentionally not fit here because LOTO validation prediction artifacts
        # are not available and FUDS test labels must not be used for gate training.
        z = (
            1.20 * (out["r5_jitter_risk"] - out["ecm_jitter_risk"])
            + 0.55 * out["r5_ood_risk"]
            + 0.35 * out["r5_uncertainty_risk"]
            - 0.85 * out["ecm_residual_risk"]
            - 0.45 * out["ecm_drift_risk"]
            - 0.18 * out["disagreement_risk"]
        )
        raw = _fixed_sigmoid(z)
    else:
        raise ValueError(f"Unknown fusion variant: {variant}")
    out["w_ecm_raw"] = np.asarray(raw, dtype=np.float64).clip(0.02, 0.98)
    out["fusion_model"] = f"{variant}_{out['ecm_model_name'].iloc[0]}"
    out = _causal_smooth_weight(out, "w_ecm_raw", "w_ecm", alpha=0.18)
    out["SOC_fused"] = out["w_ecm"] * out["SOC_ECM"] + (1.0 - out["w_ecm"]) * out["SOC_R5"]
    out["y_true"] = out["SOC_true"]
    out["y_pred"] = out["SOC_fused"]
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["model_name"] = out["fusion_model"]
    out["catastrophic_abs_error_gt_5pct"] = out["abs_error"] > 0.05
    return out


def _make_previous_label_free_variant(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["ecm_abs_voltage_residual"] = out["voltage_residual"].abs()
    out["risk_ecm_prev"] = out.groupby("experiment")["ecm_abs_voltage_residual"].transform(_robust_z)
    out["risk_disagreement_prev"] = out.groupby("experiment")["prediction_disagreement"].transform(_robust_z)
    out["w_ecm_raw"] = 1.0 / (1.0 + np.exp((out["risk_ecm_prev"] + 0.35 * out["risk_disagreement_prev"]).clip(-30, 30)))
    out["fusion_model"] = "previous_label_free_" + out["ecm_model_name"].astype(str)
    out = _causal_smooth_weight(out, "w_ecm_raw", "w_ecm", alpha=0.18)
    out["SOC_fused"] = out["w_ecm"] * out["SOC_ECM"] + (1.0 - out["w_ecm"]) * out["SOC_R5"]
    out["y_true"] = out["SOC_true"]
    out["y_pred"] = out["SOC_fused"]
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["model_name"] = out["fusion_model"]
    out["catastrophic_abs_error_gt_5pct"] = out["abs_error"] > 0.05
    return out


def _make_outside_jitter_guard_variant(base: pd.DataFrame) -> pd.DataFrame:
    out = _add_label_free_risk_scores(base)
    raw = _fixed_sigmoid(1.10 * (out["r5_risk"] - out["ecm_risk"]) - 0.10 * out["disagreement_risk"])
    r5_unstable = (out["r5_recent_jitter"] > 0.0012) | (
        out["r5_recent_jitter"] > 5.0 * out["ecm_recent_jitter"].clip(lower=1e-6)
    )
    ecm_stable = (out["ecm_residual_rms"] < 0.035) & (out["ecm_recent_jitter"] < 0.0015)
    high_disagreement = out["prediction_disagreement"] > 0.02
    raw = np.where(r5_unstable & ecm_stable, np.maximum(raw, 0.92), raw)
    raw = np.where(r5_unstable & ecm_stable & high_disagreement, np.maximum(raw, 0.97), raw)
    raw = np.where(out["ecm_residual_rms"] > 0.045, np.minimum(raw, 0.35), raw)
    out["w_ecm_raw"] = np.asarray(raw, dtype=np.float64).clip(0.02, 0.99)
    out["fusion_model"] = "outside_jitter_guard_" + out["ecm_model_name"].astype(str)
    out = _causal_smooth_weight(out, "w_ecm_raw", "w_ecm", alpha=0.25)
    out["SOC_fused"] = out["w_ecm"] * out["SOC_ECM"] + (1.0 - out["w_ecm"]) * out["SOC_R5"]
    out["y_true"] = out["SOC_true"]
    out["y_pred"] = out["SOC_fused"]
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["model_name"] = out["fusion_model"]
    out["catastrophic_abs_error_gt_5pct"] = out["abs_error"] > 0.05
    return out


def _summarize_prediction_like(df: pd.DataFrame, model_col="model_name") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, by_temp_rows, focus_rows = [], [], []
    for (exp, model), g in df.groupby(["experiment", model_col]):
        row = {"experiment": exp, "model_name": model}
        row.update(_metrics(g.rename(columns={model_col: "model_name"})))
        row["catastrophic_error_rate_gt_5pct"] = float((g["abs_error"] > 0.05).mean())
        rows.append(row)
        for temp, tg in g.groupby("temperature_C"):
            tr = {"experiment": exp, "model_name": model, "temperature_C": float(temp)}
            tr.update(_metrics(tg.rename(columns={model_col: "model_name"})))
            tr["catastrophic_error_rate_gt_5pct"] = float((tg["abs_error"] > 0.05).mean())
            if "w_ecm" in tg:
                tr["mean_w_ecm"] = float(tg["w_ecm"].mean())
            by_temp_rows.append(tr)
        omitted = omitted_temp_for(exp)
        og = g[np.isclose(g["temperature_C"].astype(float), omitted)]
        seen = g[~np.isclose(g["temperature_C"].astype(float), omitted)]
        if len(og):
            fr = {"experiment": exp, "model_name": model, "omitted_temperature_C": float(omitted)}
            for k, v in _metrics(og.rename(columns={model_col: "model_name"})).items():
                fr[f"omitted_{k}"] = v
            fr["omitted_catastrophic_error_rate_gt_5pct"] = float((og["abs_error"] > 0.05).mean())
            fr["seen_MAE_pct"] = float(seen["abs_error"].mean() * 100.0) if len(seen) else np.nan
            if "w_ecm" in og:
                fr["mean_w_ecm_omitted"] = float(og["w_ecm"].mean())
            focus_rows.append(fr)
    return pd.DataFrame(rows), pd.DataFrame(by_temp_rows), pd.DataFrame(focus_rows)


def _prepare_outside_fusion_rows(root: Path) -> pd.DataFrame:
    r5_path = root / "outside_r5_prediction_rows.csv"
    if not r5_path.exists():
        return pd.DataFrame()
    r5 = pd.read_csv(r5_path)
    if r5.empty:
        return pd.DataFrame()
    r5 = r5.rename(columns={"y_pred": "SOC_R5", "abs_error": "abs_error_R5", "error": "error_R5"}).copy()
    r5["r5_model_name"] = r5["model_name"]
    r5["model_name"] = r5["r5_model_name"]
    r5 = _causal_rolling_features(r5, "SOC_R5", window=50, prefix="r5_")
    r5 = r5.drop(columns=["model_name"])
    ecm = _read_csv(root / "neural_ecm_prediction_rows.csv")
    ecm = ecm[ecm["experiment"].isin(["Omit N10", "Omit 50"]) & ecm["model_name"].isin(ECM_FUSION_MODELS)].copy()
    if ecm.empty:
        return pd.DataFrame()
    ecm = _causal_rolling_features(ecm, "y_pred", residual_col="voltage_residual", window=50, prefix="ecm_")
    drift = _load_tta_parameter_drift(root)
    if len(drift):
        ecm = ecm.merge(drift, on=["experiment", "trajectory_id"], how="left")
    ecm["ecm_parameter_drift"] = ecm["ecm_parameter_drift"].fillna(0.0) if "ecm_parameter_drift" in ecm else 0.0
    join_cols = ["experiment", "trajectory_id", "end_index"]
    r5_cols = [
        "experiment",
        "trajectory_id",
        "end_index",
        "SOC_R5",
        "y_true",
        "temperature_C",
        "drive_cycle",
        "trajectory_fraction",
        "r5_model_name",
        "r5_recent_jitter",
    ]
    r5_small = r5[[c for c in r5_cols if c in r5.columns]].copy()
    rows = []
    for ecm_model, eg in ecm.groupby("model_name"):
        merged = eg.merge(r5_small, on=join_cols, suffixes=("_ECM", "_R5"), how="inner")
        if merged.empty:
            continue
        if "temperature_C_ECM" in merged:
            merged["temperature_C"] = merged["temperature_C_ECM"]
        elif "temperature_C_R5" in merged:
            merged["temperature_C"] = merged["temperature_C_R5"]
        if "drive_cycle_ECM" in merged:
            merged["drive_cycle"] = merged["drive_cycle_ECM"]
        elif "drive_cycle_R5" in merged:
            merged["drive_cycle"] = merged["drive_cycle_R5"]
        if "trajectory_fraction_ECM" in merged:
            merged["trajectory_fraction"] = merged["trajectory_fraction_ECM"]
        elif "trajectory_fraction_R5" in merged:
            merged["trajectory_fraction"] = merged["trajectory_fraction_R5"]
        merged["SOC_ECM"] = merged["y_pred"]
        merged["SOC_true"] = merged["y_true_ECM"] if "y_true_ECM" in merged else merged["y_true"]
        merged["ecm_model_name"] = ecm_model
        merged["prediction_disagreement"] = (merged["SOC_ECM"] - merged["SOC_R5"]).abs()
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for col, val in {
        "distance_S4_latent": 8.0,
        "mahalanobis_S3": 3.0,
        "knn_S4_k5": 3.0,
        "ensemble_std": 0.0,
        "prediction_range": 0.0,
        "gate_entropy": 0.0,
        "gate_variation": 0.0,
    }.items():
        if col not in out:
            out[col] = val
    return out


def outside_range_fusion(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _prepare_outside_fusion_rows(root)
    if base.empty:
        for name in [
            "outside_range_fusion_results.csv",
            "outside_range_fusion_by_temperature.csv",
            "outside_range_fusion_focus.csv",
        ]:
            pd.DataFrame().to_csv(root / name, index=False)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    parts = []
    for _, eg in base.groupby("ecm_model_name"):
        parts.append(_make_previous_label_free_variant(eg.copy()))
        parts.append(_make_outside_jitter_guard_variant(eg.copy()))
        for variant in ["jitter_aware_mlp_proxy", "jitter_aware_rule", "inverse_risk"]:
            parts.append(_make_fusion_variant(eg.copy(), variant))
    fused = pd.concat(parts, ignore_index=True)

    baseline_parts = []
    r5 = pd.read_csv(root / "outside_r5_prediction_rows.csv")
    r5_base = r5.copy()
    r5_base["model_name"] = "R5_GATED_AUG_REX_l1p0_outside"
    baseline_parts.append(r5_base)
    ecm = pd.read_csv(root / "neural_ecm_prediction_rows.csv")
    ecm_base = ecm[ecm["experiment"].isin(["Omit N10", "Omit 50"]) & ecm["model_name"].isin(ECM_FUSION_MODELS)].copy()
    baseline_parts.append(ecm_base)
    base_pred = pd.concat(baseline_parts, ignore_index=True, sort=False)
    if "error" not in base_pred:
        base_pred["error"] = base_pred["y_pred"] - base_pred["y_true"]
    if "abs_error" not in base_pred:
        base_pred["abs_error"] = base_pred["error"].abs()

    base_res, base_by_temp, base_focus = _summarize_prediction_like(base_pred)
    fus_res, fus_by_temp, fus_focus = _summarize_prediction_like(fused)
    results = pd.concat([base_res, fus_res], ignore_index=True, sort=False)
    by_temp = pd.concat([base_by_temp, fus_by_temp], ignore_index=True, sort=False)
    focus = pd.concat([base_focus, fus_focus], ignore_index=True, sort=False)

    for df in (results, by_temp, focus):
        if len(df):
            df["mean_voltage_residual_mae"] = np.nan
            df["mean_prediction_disagreement"] = np.nan
    if len(fused):
        diag = (
            fused.groupby(["experiment", "model_name"])
            .agg(
                mean_voltage_residual_mae=("voltage_residual", lambda s: float(np.mean(np.abs(s)))),
                mean_prediction_disagreement=("prediction_disagreement", "mean"),
                mean_w_ecm=("w_ecm", "mean"),
            )
            .reset_index()
        )
        results = results.drop(columns=[c for c in ["mean_voltage_residual_mae", "mean_prediction_disagreement", "mean_w_ecm"] if c in results], errors="ignore").merge(
            diag, on=["experiment", "model_name"], how="left"
        )
        focus = focus.drop(columns=[c for c in ["mean_voltage_residual_mae", "mean_prediction_disagreement", "mean_w_ecm"] if c in focus], errors="ignore").merge(
            diag, on=["experiment", "model_name"], how="left"
        )
    results.to_csv(root / "outside_range_fusion_results.csv", index=False)
    by_temp.to_csv(root / "outside_range_fusion_by_temperature.csv", index=False)
    focus.to_csv(root / "outside_range_fusion_focus.csv", index=False)
    fused.to_csv(root / "outside_range_fusion_prediction_rows.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for model, g in fused.groupby("model_name"):
        gg = g.groupby("temperature_C", as_index=False)["w_ecm"].mean()
        ax.plot(gg["temperature_C"], gg["w_ecm"], marker="o", label=model)
    ax.set_xlabel("outside target temperature_C")
    ax.set_ylabel("mean ECM fusion weight")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6)
    fig.savefig(root / "outside_range_fusion_weight_plot.png", dpi=170)
    plt.close(fig)
    return results, by_temp, focus


TRAIN_RANGE_BY_EXPERIMENT = {
    "Exp A": (-10.0, 50.0),
    "Exp B": (-10.0, 50.0),
    "Exp C": (-10.0, 50.0),
    "Omit N10": (0.0, 50.0),
    "Omit 50": (-10.0, 25.0),
}


def _outside_train_range_flag(df: pd.DataFrame) -> pd.Series:
    flags = []
    for exp, temp in zip(df["experiment"], df["temperature_C"].astype(float)):
        lo, hi = TRAIN_RANGE_BY_EXPERIMENT.get(str(exp), (-np.inf, np.inf))
        flags.append(bool(temp < lo or temp > hi))
    return pd.Series(flags, index=df.index)


def _hard_guard_thresholds(root: Path) -> pd.DataFrame:
    val = _prepare_jitter_fusion_rows(root)
    val = _add_label_free_risk_scores(val)
    rows = []
    for ecm_model, g in val.groupby("ecm_model_name"):
        rows.append(
            {
                "ecm_model_name": ecm_model,
                "source": "Exp A/B/C LOTO risk feature distribution; SOC labels not used",
                "r5_recent_jitter_threshold": float(g["r5_recent_jitter"].quantile(0.75)),
                "r5_disagreement_threshold": float(g["prediction_disagreement"].quantile(0.75)),
                "r5_ood_risk_threshold": float(g["r5_ood_risk"].quantile(0.75)),
                "ecm_residual_rms_high_threshold": float(2.0 * g["ecm_residual_rms"].quantile(0.99)),
                "ecm_parameter_drift_high_threshold": float(
                    max(g["ecm_parameter_drift"].quantile(0.99), 3.0 * g["ecm_parameter_drift"].quantile(0.95))
                ),
                "threshold_note": "Hard guard uses outside-range flag plus R5 jitter/disagreement/OOD risk. ECM residual uses 2*LOTO q99 and ECM drift uses max(q99, 3*q95), so fallback is reserved for extreme ECM risk without using SOC labels.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(root / "hard_guard_thresholds.csv", index=False)
    return out


def _make_hard_guard_variant(base: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ecm_model, eg in base.groupby("ecm_model_name"):
        t = thresholds[thresholds["ecm_model_name"].eq(ecm_model)]
        if t.empty:
            continue
        row = t.iloc[0]
        out = _make_fusion_variant(eg.copy(), "jitter_aware_mlp_proxy")
        out["outside_train_range"] = _outside_train_range_flag(out)
        out["r5_hard_risky"] = (
            (out["r5_recent_jitter"] > float(row["r5_recent_jitter_threshold"]))
            | (out["prediction_disagreement"] > float(row["r5_disagreement_threshold"]))
            | (out["r5_ood_risk"] > float(row["r5_ood_risk_threshold"]))
        )
        out["ecm_high_risk"] = (
            (out["ecm_residual_rms"] > float(row["ecm_residual_rms_high_threshold"]))
            | (out["ecm_parameter_drift"] > float(row["ecm_parameter_drift_high_threshold"]))
        )
        out["hard_ecm_override"] = out["outside_train_range"] & out["r5_hard_risky"] & ~out["ecm_high_risk"]
        out["hard_fallback"] = out["outside_train_range"] & out["ecm_high_risk"]
        out.loc[out["hard_ecm_override"], "w_ecm"] = 1.0
        out.loc[out["hard_fallback"], "w_ecm"] = np.minimum(out.loc[out["hard_fallback"], "w_ecm"], 0.45)
        out["SOC_fused"] = out["w_ecm"] * out["SOC_ECM"] + (1.0 - out["w_ecm"]) * out["SOC_R5"]
        out["y_pred"] = out["SOC_fused"]
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = out["error"].abs()
        out["model_name"] = "hard_guard_" + ecm_model
        out["fusion_model"] = out["model_name"]
        out["catastrophic_abs_error_gt_5pct"] = out["abs_error"] > 0.05
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def hard_guard_fusion(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = _hard_guard_thresholds(root)
    val_base = _prepare_jitter_fusion_rows(root)
    outside_base = _prepare_outside_fusion_rows(root)
    base = pd.concat([val_base, outside_base], ignore_index=True, sort=False)
    hard = _make_hard_guard_variant(base, thresholds)
    hard_res, hard_by_temp, hard_focus = _summarize_prediction_like(hard) if len(hard) else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    focus_parts = []
    jf = root / "jitter_aware_fusion_omitted_focus.csv"
    of = root / "outside_range_fusion_focus.csv"
    if jf.exists():
        focus_parts.append(pd.read_csv(jf))
    if of.exists():
        focus_parts.append(pd.read_csv(of))
    if focus_parts:
        baseline_focus = pd.concat(focus_parts, ignore_index=True, sort=False)
        keep_names = {
            "R5_GATED_AUG_REX_best_per_fold",
            "NeuralECMObserver_REX",
            "NeuralECMObserver_REX_TTA_voltage_only",
            "previous_Fusion_label_free_NeuralECMObserver_REX_TTA_voltage_only",
            "jitter_aware_mlp_proxy_NeuralECMObserver_REX",
            "jitter_aware_rule_NeuralECMObserver_REX_TTA_voltage_only",
            "R5_GATED_AUG_REX_l1p0_outside",
            "previous_label_free_NeuralECMObserver_REX_TTA_voltage_only",
            "outside_jitter_guard_NeuralECMObserver_REX_TTA_voltage_only",
        }
        baseline_focus = baseline_focus[baseline_focus["model_name"].isin(keep_names)].copy()
        focus = pd.concat([baseline_focus, hard_focus], ignore_index=True, sort=False)
    else:
        focus = hard_focus

    hard_res.to_csv(root / "hard_guard_fusion_results.csv", index=False)
    hard_by_temp.to_csv(root / "hard_guard_fusion_by_temperature.csv", index=False)
    focus.to_csv(root / "hard_guard_fusion_focus.csv", index=False)
    if len(hard):
        cols = [
            "experiment",
            "model_name",
            "ecm_model_name",
            "trajectory_id",
            "end_index",
            "temperature_C",
            "outside_train_range",
            "r5_hard_risky",
            "ecm_high_risk",
            "hard_ecm_override",
            "hard_fallback",
            "w_ecm",
            "r5_recent_jitter",
            "prediction_disagreement",
            "r5_ood_risk",
            "ecm_residual_rms",
            "ecm_parameter_drift",
            "abs_error",
        ]
        hard[[c for c in cols if c in hard.columns]].to_csv(root / "hard_guard_fusion_diagnostics.csv", index=False)
    _write_hard_guard_report(root, thresholds, focus)
    return hard_res, hard_by_temp, focus


def _load_target_cycle_frames(
    cfg: CFG,
    *,
    target_temp: float,
    train_temps: tuple[str, ...],
    test_drives=("DST", "US06"),
) -> dict[str, list[pd.DataFrame]]:
    copied = _ensure_target_cycle_feature_cache(
        cfg,
        target_temp=target_temp,
        train_temps=train_temps,
        test_drives=test_drives,
    )
    if copied:
        pd.DataFrame(copied).to_csv(
            _repo_root(cfg) / "dst_us06_feature_cache_fill_log.csv",
            mode="a",
            header=not (_repo_root(cfg) / "dst_us06_feature_cache_fill_log.csv").exists(),
            index=False,
        )
    train_temp_c = {temp_key_to_float(t) for t in train_temps}
    test_drives_u = {str(d).upper() for d in test_drives}
    out = {"train": [], "valid": [], "test": []}
    for path in sorted(Path(cfg.decomposed_dir).glob("*_features.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        temp = float(frame["temperature"].iloc[0])
        drive = str(frame["drive_cycle"].iloc[0]).upper()
        if drive in {"DST", "US06"} and temp in train_temp_c:
            out["train"].append(frame)
        if drive in test_drives_u and np.isclose(temp, float(target_temp)):
            out["test"].append(frame)
    train_ids = {f["trajectory_id"].iloc[0] for f in out["train"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in out["test"]}
    assert train_ids.isdisjoint(test_ids), "Train/test leakage in target-cycle extrapolation"
    if not out["train"]:
        raise ValueError("No train frames for target-cycle extrapolation")
    return out


def _temp_key_from_float(temp: float) -> str:
    t = float(temp)
    if np.isclose(t, -10.0):
        return "N10"
    if abs(t - round(t)) < 1e-6:
        return str(int(round(t)))
    return f"{t:g}".replace("-", "N")


def _ensure_target_cycle_feature_cache(
    cfg: CFG,
    *,
    target_temp: float,
    train_temps: tuple[str, ...],
    test_drives=("DST", "US06"),
) -> list[dict]:
    """Fill missing target-cycle feature CSVs from existing feature caches.

    Raw DST/US06 CSVs exist in the data directory.  This helper deliberately
    does not fabricate decomposed features; it only reuses feature files that
    were already generated by a corrector experiment, so comparisons stay tied
    to learned voltage decomposition features.
    """
    dst = Path(cfg.decomposed_dir)
    dst.mkdir(parents=True, exist_ok=True)
    root = _repo_root(cfg)
    candidate_dirs = [p for p in sorted(root.glob("decomposed_features*")) if p.is_dir()]
    target_key = _temp_key_from_float(target_temp)
    required = []
    for temp_key in train_temps:
        for drive in ("DST", "US06"):
            required.append(f"LFP_{temp_key}_{drive}_features.csv")
    for drive in test_drives:
        required.append(f"LFP_{target_key}_{str(drive).upper()}_features.csv")

    copied = []
    for name in sorted(set(required)):
        dest = dst / name
        if dest.exists():
            continue
        src = next((d / name for d in candidate_dirs if (d / name).exists()), None)
        if src is None:
            copied.append({
                "target_dir": str(dst),
                "feature_file": name,
                "status": "missing",
                "source": "",
                "note": "No generated decomposed feature cache found. Generate it from the raw CSV with the corrector pipeline before this comparison.",
            })
            continue
        shutil.copy2(src, dest)
        copied.append({
            "target_dir": str(dst),
            "feature_file": name,
            "status": "copied",
            "source": str(src),
            "note": "Copied from an existing generated decomposed feature cache.",
        })
    missing = [r["feature_file"] for r in copied if r["status"] == "missing"]
    if missing:
        raise FileNotFoundError(
            "Missing decomposed feature CSVs for DST/US06 target-cycle extrapolation:\n"
            + "\n".join(missing)
            + "\nRaw CSVs may exist, but decomposed-feature experiments need the corrector-generated *_features.csv cache."
        )
    return copied


def temp_key_to_float(v) -> float:
    s = str(v).strip().upper()
    return -float(s[1:]) if s.startswith("N") else float(s)


TARGET_CYCLE_EXPERIMENTS = {
    "Exp A": {
        "train_temps": ("N10", "0", "25", "50"),
        "target_temperature_C": 10.0,
        "feature_dir": "decomposed_features",
    },
    "Exp B": {
        "train_temps": ("N10", "10", "25", "50"),
        "target_temperature_C": 0.0,
        "feature_dir": "decomposed_features",
    },
    "Exp C": {
        "train_temps": ("N10", "0", "10", "25", "50"),
        "target_temperature_C": 20.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_20_25_50",
    },
    "Omit N10": {
        "train_temps": ("0", "10", "25", "50"),
        "target_temperature_C": -10.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
    },
    "Omit 50": {
        "train_temps": ("N10", "0", "10", "25"),
        "target_temperature_C": 50.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
    },
}


def target_cycle_cfg(cfg: CFG | None, experiment: str) -> CFG:
    base = ecm_exp_cfg(cfg or make_cfg(), experiment) if experiment in ("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50") else make_cfg()
    spec = TARGET_CYCLE_EXPERIMENTS[experiment]
    base.train_temps = spec["train_temps"]
    base.train_drives = ("DST", "US06")
    base.eval_drive = "DST_US06_TARGET"
    base.decomposed_dir = base.output_dir / spec["feature_dir"]
    return base


def _target_cycle_prefix(experiment: str) -> str:
    return experiment.replace(" ", "_").replace("-", "m")


def _merge_for_target_cycle_fusion(
    r5_pred: pd.DataFrame,
    ecm_pred: pd.DataFrame,
    *,
    experiment: str,
) -> pd.DataFrame:
    r5 = r5_pred.rename(columns={"y_pred": "SOC_R5", "error": "error_R5", "abs_error": "abs_error_R5"}).copy()
    r5["r5_model_name"] = r5["model_name"]
    r5["model_name"] = r5["r5_model_name"]
    r5 = _causal_rolling_features(r5, "SOC_R5", window=50, prefix="r5_").drop(columns=["model_name"])
    ecm = _causal_rolling_features(ecm_pred, "y_pred", residual_col="voltage_residual", window=50, prefix="ecm_")
    rows = []
    join_cols = ["experiment", "trajectory_id", "end_index"]
    r5_cols = [
        "experiment", "trajectory_id", "end_index", "SOC_R5", "y_true",
        "temperature_C", "drive_cycle", "trajectory_fraction", "r5_model_name", "r5_recent_jitter",
    ]
    for ecm_model, eg in ecm.groupby("model_name"):
        merged = eg.merge(r5[[c for c in r5_cols if c in r5]], on=join_cols, suffixes=("_ECM", "_R5"), how="inner")
        if merged.empty:
            continue
        if "temperature_C_ECM" in merged:
            merged["temperature_C"] = merged["temperature_C_ECM"]
        elif "temperature_C_R5" in merged:
            merged["temperature_C"] = merged["temperature_C_R5"]
        if "drive_cycle_ECM" in merged:
            merged["drive_cycle"] = merged["drive_cycle_ECM"]
        elif "drive_cycle_R5" in merged:
            merged["drive_cycle"] = merged["drive_cycle_R5"]
        if "trajectory_fraction_ECM" in merged:
            merged["trajectory_fraction"] = merged["trajectory_fraction_ECM"]
        elif "trajectory_fraction_R5" in merged:
            merged["trajectory_fraction"] = merged["trajectory_fraction_R5"]
        merged["SOC_ECM"] = merged["y_pred"]
        merged["SOC_true"] = merged["y_true_ECM"] if "y_true_ECM" in merged else merged["y_true"]
        merged["ecm_model_name"] = ecm_model
        merged["prediction_disagreement"] = (merged["SOC_ECM"] - merged["SOC_R5"]).abs()
        merged["ecm_parameter_drift"] = merged.get("ecm_parameter_drift", 0.0)
        for col, val in {
            "distance_S4_latent": 8.0,
            "mahalanobis_S3": 3.0,
            "knn_S4_k5": 3.0,
            "ensemble_std": 0.0,
            "prediction_range": 0.0,
            "gate_entropy": 0.0,
            "gate_variation": 0.0,
        }.items():
            merged[col] = val
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_dst_us06_target_cycle_extrapolation(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50"),
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    configure_torch_runtime()
    cfg = cfg or make_cfg()
    root = _repo_root(cfg)
    out_path = root / "dst_us06_extrapolation_focus.csv"
    if out_path.exists() and not force:
        return {
            "focus": pd.read_csv(out_path),
            "results": pd.read_csv(root / "dst_us06_extrapolation_results.csv") if (root / "dst_us06_extrapolation_results.csv").exists() else pd.DataFrame(),
        }
    all_pred, all_results, all_by_temp, all_focus, all_notes = [], [], [], [], []
    thresholds = _hard_guard_thresholds(root)
    for experiment in experiments:
        spec = TARGET_CYCLE_EXPERIMENTS[experiment]
        ecfg = target_cycle_cfg(cfg, experiment)
        target_temp = float(spec["target_temperature_C"])
        feature_frames = _load_target_cycle_frames(
            ecfg,
            target_temp=target_temp,
            train_temps=spec["train_temps"],
            test_drives=("DST", "US06"),
        )
        if not feature_frames["test"]:
            all_notes.append({
                "experiment": experiment,
                "target_temperature_C": target_temp,
                "status": "skipped",
                "reason": "No DST/US06 target-temperature feature files are available.",
            })
            continue
        all_notes.append({
            "experiment": experiment,
            "target_temperature_C": target_temp,
            "status": "completed",
            "reason": "",
            "n_train_trajectories": len(feature_frames["train"]),
            "n_test_trajectories": len(feature_frames["test"]),
            "test_trajectory_ids": ",".join(str(f["trajectory_id"].iloc[0]) for f in feature_frames["test"]),
        })

        r5_name = "R5_GATED_AUG_REX_l1p0_DST_US06_target"
        _, r5_hist, _, r5_raw = train_rex_model(
            feature_frames,
            ecfg,
            r5_name,
            lambda_rex=1.0,
            use_aug=True,
            experiment=experiment,
        )
        r5_attached, r5_res, r5_by_temp, r5_focus = attach_and_summarize(
            [(r5_name, r5_raw)],
            feature_frames,
            ecfg,
            experiment,
            target_temp,
        )
        all_pred.append(r5_attached)
        all_results.append(r5_res)
        all_by_temp.append(r5_by_temp)
        all_focus.append(r5_focus)
        r5_hist.to_csv(root / f"dst_us06_{_target_cycle_prefix(experiment)}_r5_history.csv", index=False)

        ecm_spec = ECMSpec(
            name="NeuralECMObserver_REX_DST_US06_target",
            use_rex=True,
            lambda_v=0.20,
            lambda_rex=0.5,
            lambda_worst=0.10,
            correction_limit=0.05,
        )
        ecm_model, ecm_hist, _ = train_ecm_model(feature_frames, ecfg, ecm_spec, experiment)
        ecm_hist.to_csv(root / f"dst_us06_{_target_cycle_prefix(experiment)}_ecm_history.csv", index=False)
        ecm_pred = predict_full_trajectories(ecm_model, feature_frames["test"], ecm_spec.name)
        ecm_pred["experiment"] = experiment
        scales = {}
        scale_rows = []
        for frame in feature_frames["test"]:
            tid = str(frame["trajectory_id"].iloc[0])
            s = adapt_voltage_only(ecm_model, frame, ecfg, steps=int(getattr(ecfg, "ecm_tta_steps", 12)))
            scales[tid] = s
            scale_rows.append({
                "experiment": experiment,
                "trajectory_id": tid,
                "temperature_C": float(frame["temperature"].iloc[0]),
                "log_Q_scale": float(s[0].cpu()),
                "log_R0_scale": float(s[1].cpu()),
                "log_RC_gain_scale": float(s[2].cpu()),
                "log_tau_scale": float(s[3].cpu()),
                "log_hys_gain_scale": float(s[4].cpu()),
            })
        scale_df = pd.DataFrame(scale_rows)
        scale_df.to_csv(root / f"dst_us06_{_target_cycle_prefix(experiment)}_ecm_tta_scales.csv", index=False)
        ecm_tta = predict_full_trajectories(
            ecm_model,
            feature_frames["test"],
            "NeuralECMObserver_REX_TTA_DST_US06_target",
            scales_by_tid=scales,
        )
        ecm_tta["experiment"] = experiment
        for frame_pred in [ecm_pred, ecm_tta]:
            lookup = build_prediction_feature_lookup(feature_frames)
            attached = attach_prediction_features(
                frame_pred.assign(split="test", ablation=frame_pred["model_name"].iloc[0]),
                lookup,
                ablation_name=frame_pred["model_name"].iloc[0],
                target_label="physical",
            )
            attached["experiment"] = experiment
            all_pred.append(attached)
            res = _overall_metrics(attached)
            res["experiment"] = experiment
            bt = variance_by_temperature(attached)
            bt["experiment"] = experiment
            _, _, focus = _summarize_prediction_like(attached)
            all_results.append(res)
            all_by_temp.append(bt)
            all_focus.append(focus)

        base_fusion = _merge_for_target_cycle_fusion(r5_attached, pd.concat([ecm_pred, ecm_tta], ignore_index=True), experiment=experiment)
        if len(base_fusion):
            fusion_parts = []
            for _, eg in base_fusion.groupby("ecm_model_name"):
                fusion_parts.append(_make_previous_label_free_variant(eg.copy()))
                fusion_parts.append(_make_fusion_variant(eg.copy(), "jitter_aware_mlp_proxy"))
                fusion_parts.append(_make_fusion_variant(eg.copy(), "jitter_aware_rule"))
                fusion_parts.append(_make_hard_guard_variant(eg.copy(), thresholds))
            fused = pd.concat(fusion_parts, ignore_index=True)
            all_pred.append(fused)
            fr, fbt, ff = _summarize_prediction_like(fused)
            all_results.append(fr)
            all_by_temp.append(fbt)
            all_focus.append(ff)

    pred = pd.concat(all_pred, ignore_index=True, sort=False) if all_pred else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True, sort=False) if all_results else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True, sort=False) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True, sort=False) if all_focus else pd.DataFrame()
    notes = pd.DataFrame(all_notes)
    pred.to_csv(root / "dst_us06_extrapolation_prediction_rows.csv", index=False)
    results.to_csv(root / "dst_us06_extrapolation_results.csv", index=False)
    by_temp.to_csv(root / "dst_us06_extrapolation_by_temperature.csv", index=False)
    focus.to_csv(root / "dst_us06_extrapolation_focus.csv", index=False)
    notes.to_csv(root / "dst_us06_extrapolation_notes.csv", index=False)
    write_all_experiment_summary(root)
    return {"predictions": pred, "results": results, "by_temperature": by_temp, "focus": focus, "notes": notes}


def _write_hard_guard_report(root: Path, thresholds: pd.DataFrame, focus: pd.DataFrame) -> None:
    lines = ["# Hard Risk-Guarded Fusion Report\n"]
    lines.append("Thresholds are selected from Exp A/B/C LOTO risk-feature distributions only. SOC labels are not used to choose hard-guard thresholds.\n")
    lines.append("Hard rule: if a sample is outside the train temperature range and R5 jitter, disagreement, or OOD risk exceeds the LOTO threshold, set w_ECM=1.0 unless ECM residual/drift is high.\n")
    lines.append("If ECM residual or drift is high, the model falls back to jitter-aware soft fusion/R5 contribution.\n")
    lines.append("\n## Thresholds\n")
    lines.append(thresholds.to_markdown(index=False, floatfmt=".6f"))
    lines.append("\n\n## Focus Metrics\n")
    if len(focus):
        cols = [
            "experiment",
            "model_name",
            "omitted_temperature_C",
            "omitted_MAE_pct",
            "omitted_RMSE_pct",
            "omitted_jitter_ratio",
            "omitted_catastrophic_error_rate_gt_5pct",
            "mean_w_ecm",
        ]
        keep = focus[[c for c in cols if c in focus.columns]].dropna(subset=["omitted_MAE_pct"])
        lines.append(keep.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\n## Interpretation\n")
    lines.append("- If outside -10C approaches NeuralECM_TTA, the hard guard successfully suppresses catastrophic R5 contribution.\n")
    lines.append("- If omitted 0/10/20C metrics stay close to jitter-aware fusion, the guard does not harm in-range omitted folds.\n")
    lines.append("- If outside 50C remains high, high-temperature outside-range extrapolation remains unresolved.\n")
    lines.append("- TTA rows remain voltage-only transductive adaptation, not pure extrapolation.\n")
    lines.append("\nForbidden interpretation: hard guard guarantees robustness at all unseen temperatures.\n")
    (root / "hard_guard_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_all_experiment_summary(root: Path) -> Path:
    sections = ["# All Robustness Experiments Summary\n"]
    sections.append("This file consolidates the FUDS-based temperature extrapolation diagnostics and the new DST/US06 target-cycle extrapolation check. DST/US06 target-cycle evaluation uses only omitted/outside target temperatures, because DST/US06 at train temperatures are already supervised train trajectories.\n")
    files = [
        ("FUDS REx omitted-temperature", "rex_omitted_temp_focus.csv"),
        ("FUDS NeuralECM omitted/outside", "neural_ecm_omitted_temp_focus.csv"),
        ("FUDS jitter-aware fusion", "jitter_aware_fusion_omitted_focus.csv"),
        ("FUDS hard-guard fusion", "hard_guard_fusion_focus.csv"),
        ("DST/US06 target-cycle extrapolation", "dst_us06_extrapolation_focus.csv"),
        ("DST/US06 target-cycle notes", "dst_us06_extrapolation_notes.csv"),
    ]
    for title, name in files:
        path = root / name
        sections.append(f"\n## {title}\n")
        if not path.exists():
            sections.append(f"`{name}` not available.\n")
            continue
        df = pd.read_csv(path)
        if df.empty:
            sections.append("No rows.\n")
            continue
        if "omitted_MAE_pct" in df.columns:
            cols = [
                "experiment",
                "model_name",
                "omitted_temperature_C",
                "omitted_MAE_pct",
                "omitted_RMSE_pct",
                "omitted_jitter_ratio",
                "omitted_catastrophic_error_rate_gt_5pct",
                "mean_w_ecm",
            ]
            keep = df[[c for c in cols if c in df.columns]].dropna(subset=["omitted_MAE_pct"], how="all")
            if len(keep) > 80:
                keep = keep.head(80)
            sections.append(keep.to_markdown(index=False, floatfmt=".3f"))
            sections.append("\n")
        else:
            sections.append(df.to_markdown(index=False))
            sections.append("\n")
    sections.append("\n## Key Interpretation\n")
    sections.append("- The earlier extrapolation results were primarily `train=DST+US06`, `test=FUDS`, so temperature omission and drive-cycle transfer were coupled.\n")
    sections.append("- DST/US06 target-cycle re-evaluation avoids leakage by testing only target temperatures omitted from the DST/US06 training set.\n")
    sections.append("- Exp C omitted 20C was repeated on DST/US06 using `LFP_20_DST` and `LFP_20_US06` generated decomposed feature files.\n")
    sections.append("- Voltage-only TTA remains transductive adaptation, not pure extrapolation.\n")
    sections.append("- Pure unseen-temperature extrapolation is not solved unless results improve consistently without target-trajectory adaptation.\n")
    sections.append("\n## Safe Claims\n")
    sections.append("- Jitter-aware and hard-guard fusion reduce temporal instability or suppress catastrophic R5 contribution in specific risk regimes.\n")
    sections.append("- R5 and NeuralECM have complementary failure modes.\n")
    sections.append("- Representative temperature coverage remains necessary.\n")
    sections.append("\n## Forbidden Claims\n")
    sections.append("- The model solves pure temperature extrapolation.\n")
    sections.append("- FUDS TTA or DST/US06 TTA is pure extrapolation.\n")
    sections.append("- Learned voltage components are true physical polarization or hysteresis.\n")
    path = root / "all_experiment_results_summary.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def _baseline_rows_for_jitter_fusion(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_parts = []
    r5 = _prepare_r5_for_fusion(root)
    if len(r5):
        r5_base = r5.rename(columns={"SOC_R5": "y_pred", "error_R5": "error", "abs_error_R5": "abs_error"}).copy()
        r5_base["source_model_name"] = [
            BEST_R5_AUG_REX_BY_EXP.get(exp, "R5_GATED_AUG_REX") for exp in r5_base["experiment"]
        ]
        r5_base["model_name"] = "R5_GATED_AUG_REX_best_per_fold"
        r5_base = r5_base.loc[:, ~r5_base.columns.duplicated()].copy()
        pred_parts.append(r5_base)
    ecm = _prepare_ecm_for_fusion(root)
    if len(ecm):
        ecm_base = ecm.rename(columns={"y_pred": "y_pred"}).copy()
        pred_parts.append(ecm_base)
    prev_path = root / "dual_estimator_fusion_prediction_rows.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        if len(prev):
            keep = prev.copy()
            if "SOC_fused" in keep.columns:
                keep["y_pred"] = keep["SOC_fused"]
            if "SOC_true" in keep.columns:
                keep["y_true"] = keep["SOC_true"]
            keep["model_name"] = "previous_" + keep["model_name"].astype(str)
            keep = keep.loc[:, ~keep.columns.duplicated()].copy()
            pred_parts.append(keep)
    if not pred_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = pd.concat(pred_parts, ignore_index=True, sort=False)
    if "temperature_C" not in base:
        if "temperature_C_ECM" in base:
            base["temperature_C"] = base["temperature_C_ECM"]
        elif "temperature_C_R5" in base:
            base["temperature_C"] = base["temperature_C_R5"]
    if "y_true" not in base and "SOC_true" in base:
        base["y_true"] = base["SOC_true"]
    if "error" not in base or base["error"].isna().all():
        base["error"] = base["y_pred"] - base["y_true"]
    if "abs_error" not in base or base["abs_error"].isna().all():
        base["abs_error"] = base["error"].abs()
    base = base[base["experiment"].isin(["Exp A", "Exp B", "Exp C"])].copy()
    return _summarize_prediction_like(base)


def jitter_aware_fusion(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _prepare_jitter_fusion_rows(root)
    fusion_parts = []
    if not base.empty:
        for ecm_model, eg in base.groupby("ecm_model_name"):
            for variant in ["jitter_aware_rule", "inverse_risk", "jitter_aware_mlp_proxy"]:
                fusion_parts.append(_make_fusion_variant(eg.copy(), variant))
    fused = pd.concat(fusion_parts, ignore_index=True) if fusion_parts else pd.DataFrame()
    base_res, base_by_temp, base_focus = _baseline_rows_for_jitter_fusion(root)
    fus_res, fus_by_temp, fus_focus = _summarize_prediction_like(fused) if len(fused) else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    results = pd.concat([base_res, fus_res], ignore_index=True, sort=False)
    by_temp = pd.concat([base_by_temp, fus_by_temp], ignore_index=True, sort=False)
    focus = pd.concat([base_focus, fus_focus], ignore_index=True, sort=False)

    outside_rows = []
    outside_focus_path = root / "neural_ecm_omitted_temp_focus.csv"
    if outside_focus_path.exists():
        nf = pd.read_csv(outside_focus_path)
        for exp in ["Omit N10", "Omit 50"]:
            g = nf[nf["experiment"].eq(exp) & nf["model_name"].isin(list(ECM_FUSION_MODELS))]
            for _, row in g.iterrows():
                outside_rows.append(
                    {
                        "experiment": exp,
                        "model_name": row["model_name"],
                        "omitted_temperature_C": row["omitted_temperature_C"],
                        "omitted_MAE_pct": row["omitted_MAE_pct"],
                        "omitted_RMSE_pct": row["omitted_RMSE_pct"],
                        "omitted_jitter_ratio": row["omitted_jitter_ratio"],
                        "fusion_available": False,
                        "note": "R5 outside-range prediction artifact is not available; fusion not evaluated for this fold.",
                    }
                )
    outside_df = pd.DataFrame(outside_rows)
    if len(outside_df):
        focus = pd.concat([focus, outside_df], ignore_index=True, sort=False)

    results.to_csv(root / "jitter_aware_fusion_results.csv", index=False)
    by_temp.to_csv(root / "jitter_aware_fusion_by_temperature.csv", index=False)
    focus.to_csv(root / "jitter_aware_fusion_omitted_focus.csv", index=False)
    if len(fused):
        weight_cols = [
            "experiment",
            "fusion_model",
            "ecm_model_name",
            "trajectory_id",
            "end_index",
            "temperature_C",
            "w_ecm",
            "w_ecm_raw",
            "ecm_residual_rms",
            "ecm_parameter_drift",
            "prediction_disagreement",
            "r5_recent_jitter",
            "ecm_recent_jitter",
            "distance_S4_latent",
            "mahalanobis_S3",
            "ensemble_std",
            "gate_entropy",
            "r5_risk",
            "ecm_risk",
            "disagreement_risk",
        ]
        fused[[c for c in weight_cols if c in fused.columns]].to_csv(root / "fusion_weight_diagnostics.csv", index=False)
        fused.to_csv(root / "jitter_aware_fusion_prediction_rows.csv", index=False)
    else:
        pd.DataFrame().to_csv(root / "fusion_weight_diagnostics.csv", index=False)

    if len(fus_by_temp):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model, g in fus_by_temp.groupby("model_name"):
            if "mean_w_ecm" not in g:
                continue
            gg = g.groupby("temperature_C", as_index=False)["mean_w_ecm"].mean()
            ax.plot(gg["temperature_C"], gg["mean_w_ecm"], marker="o", label=model)
        ax.set_xlabel("temperature_C")
        ax.set_ylabel("mean ECM fusion weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=1)
        fig.savefig(root / "fusion_weight_by_temperature.png", dpi=170)
        plt.close(fig)

    if len(fused):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        sc = axes[0].scatter(
            fused["ecm_residual_rms"] * 1000.0,
            fused["w_ecm"],
            c=fused["temperature_C"],
            s=3,
            alpha=0.22,
            cmap="viridis",
        )
        axes[0].set_xlabel("ECM causal residual RMS (mV)")
        axes[0].set_ylabel("ECM fusion weight")
        axes[0].grid(True, alpha=0.25)
        fig.colorbar(sc, ax=axes[0], label="temperature_C")
        axes[1].scatter(fused["prediction_disagreement"] * 100.0, fused["w_ecm"], s=3, alpha=0.22)
        axes[1].set_xlabel("|R5 - ECM| disagreement (%SOC)")
        axes[1].set_ylabel("ECM fusion weight")
        axes[1].grid(True, alpha=0.25)
        fig.savefig(root / "fusion_weight_vs_residual_disagreement.png", dpi=170)
        plt.close(fig)

    _write_fusion_jitter_report(root, results, focus)
    return results, by_temp, focus


def _write_fusion_jitter_report(root: Path, results: pd.DataFrame, focus: pd.DataFrame) -> None:
    report = ["# Jitter-Aware Fusion Report\n\n"]
    report.append("Fusion weights are label-free: test SOC labels are used only for evaluation metrics, not for gate decisions.\n")
    report.append("The gate uses causal local jitter, ECM voltage residual, parameter drift, R5/ECM disagreement, and available R5 OOD/uncertainty proxies.\n")
    report.append("Supervised MLP fusion was not trained because LOTO validation prediction artifacts were not available; fitting it on FUDS test labels would be leakage. The `jitter_aware_mlp_proxy` row is a fixed label-free nonlinear proxy, not a supervised MLP.\n\n")
    focus_core = focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])].copy()
    if len(focus_core):
        summary_rows = []
        for model, g in focus_core.groupby("model_name"):
            mae_col = "omitted_MAE_pct"
            rmse_col = "omitted_RMSE_pct"
            jit_col = "omitted_jitter_ratio"
            if mae_col not in g:
                continue
            summary_rows.append(
                {
                    "model_name": model,
                    "avg_omitted_MAE_pct": float(g[mae_col].mean()),
                    "worst_omitted_MAE_pct": float(g[mae_col].max()),
                    "avg_omitted_RMSE_pct": float(g[rmse_col].mean()) if rmse_col in g else np.nan,
                    "worst_omitted_RMSE_pct": float(g[rmse_col].max()) if rmse_col in g else np.nan,
                    "avg_jitter_ratio": float(g[jit_col].mean()) if jit_col in g else np.nan,
                    "worst_jitter_ratio": float(g[jit_col].max()) if jit_col in g else np.nan,
                }
            )
        summary = pd.DataFrame(summary_rows).sort_values("avg_omitted_MAE_pct")
        report.append("## Omitted Fold Summary\n\n")
        report.append(summary.to_markdown(index=False, floatfmt=".3f"))
        report.append("\n\n")
    outside = focus[focus["experiment"].isin(["Omit N10", "Omit 50"])].copy()
    if len(outside):
        report.append("## Outside-Range Note\n\n")
        report.append("Outside-range fusion was not evaluated because matching R5 outside-range prediction artifacts are not available. Existing NeuralECM outside metrics are listed in `jitter_aware_fusion_omitted_focus.csv` for reference.\n\n")
    report.append("## Interpretation\n\n")
    report.append("- If a fusion row lowers average/worst MAE but keeps high jitter, it should be interpreted as risk fusion, not temporal stabilization.\n")
    report.append("- If jitter-aware rows reduce jitter while preserving MAE, that is evidence that the gate is helping temporal stability.\n")
    report.append("- High-temperature outside-range extrapolation remains unresolved unless an outside 50C fusion run with a proper R5 outside model also improves.\n")
    report.append("- TTA rows remain transductive voltage-only adaptation and must be separated from pure extrapolation.\n")
    report.append("\nForbidden interpretation: pure unseen-temperature extrapolation is solved.\n")
    (root / "fusion_jitter_report.md").write_text("".join(report), encoding="utf-8")


def _make_proxy_with_params(base: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = _add_label_free_risk_scores(base)
    z = (
        float(params["jitter_weight"]) * (out["r5_jitter_risk"] - out["ecm_jitter_risk"])
        + 0.55 * out["r5_ood_risk"]
        + 0.35 * out["r5_uncertainty_risk"]
        - float(params["residual_weight"]) * out["ecm_residual_risk"]
        - 0.45 * out["ecm_drift_risk"]
        - float(params["disagreement_weight"]) * out["disagreement_risk"]
        + float(params.get("temperature_weight", 0.0)) * ((out["temperature_C"].astype(float) - 20.0).abs() / 40.0)
    )
    out["w_ecm_raw"] = _fixed_sigmoid(float(params["sharpness"]) * z).clip(0.02, 0.98)
    out["fusion_model"] = "sensitivity_proxy_" + out["ecm_model_name"].astype(str)
    out = _causal_smooth_weight(out, "w_ecm_raw", "w_ecm", alpha=0.18)
    out["SOC_fused"] = out["w_ecm"] * out["SOC_ECM"] + (1.0 - out["w_ecm"]) * out["SOC_R5"]
    out["y_true"] = out["SOC_true"]
    out["y_pred"] = out["SOC_fused"]
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["model_name"] = out["fusion_model"]
    return out


def fusion_proxy_sensitivity(root: Path) -> pd.DataFrame:
    base = _prepare_jitter_fusion_rows(root)
    if base.empty:
        out = pd.DataFrame()
        out.to_csv(root / "fusion_proxy_sensitivity.csv", index=False)
        return out
    grids = []
    for residual_weight in [0.65, 0.85, 1.05]:
        for disagreement_weight in [0.08, 0.18, 0.32]:
            for jitter_weight in [0.9, 1.2, 1.5]:
                for sharpness in [0.75, 1.0, 1.25]:
                    grids.append(
                        {
                            "residual_weight": residual_weight,
                            "disagreement_weight": disagreement_weight,
                            "jitter_weight": jitter_weight,
                            "temperature_weight": 0.0,
                            "sharpness": sharpness,
                        }
                    )
    rows = []
    for i, params in enumerate(grids):
        for ecm_model, eg in base.groupby("ecm_model_name"):
            pred = _make_proxy_with_params(eg.copy(), params)
            _, _, focus = _summarize_prediction_like(pred)
            vals = focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])].dropna(subset=["omitted_MAE_pct"])
            if vals.empty:
                continue
            row = {"sensitivity_id": i, "ecm_model_name": ecm_model, **params}
            row.update(
                {
                    "avg_omitted_MAE_pct": float(vals["omitted_MAE_pct"].mean()),
                    "worst_omitted_MAE_pct": float(vals["omitted_MAE_pct"].max()),
                    "avg_omitted_RMSE_pct": float(vals["omitted_RMSE_pct"].mean()),
                    "worst_omitted_RMSE_pct": float(vals["omitted_RMSE_pct"].max()),
                    "avg_jitter_ratio": float(vals["omitted_jitter_ratio"].mean()),
                    "worst_jitter_ratio": float(vals["omitted_jitter_ratio"].max()),
                    "avg_catastrophic_error_rate_gt_5pct": float(vals["omitted_catastrophic_error_rate_gt_5pct"].mean()),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(root / "fusion_proxy_sensitivity.csv", index=False)
    if len(out):
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        sc = ax.scatter(
            out["avg_omitted_MAE_pct"],
            out["avg_jitter_ratio"],
            c=out["worst_omitted_MAE_pct"],
            s=24,
            alpha=0.75,
            cmap="magma",
        )
        ax.set_xlabel("avg omitted MAE (%)")
        ax.set_ylabel("avg jitter ratio")
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, label="worst omitted MAE (%)")
        fig.savefig(root / "fusion_proxy_sensitivity_plot.png", dpi=170)
        plt.close(fig)
    return out


def fusion_pareto(root: Path) -> pd.DataFrame:
    focus = _read_csv(root / "jitter_aware_fusion_omitted_focus.csv")
    core = focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])].copy()
    wanted = [
        "R5_GATED_AUG_REX_best_per_fold",
        "NeuralECMObserver_REX",
        "NeuralECMObserver_REX_TTA_voltage_only",
        "previous_Fusion_label_free_NeuralECMObserver_REX",
        "previous_Fusion_label_free_NeuralECMObserver_REX_TTA_voltage_only",
        "jitter_aware_mlp_proxy_NeuralECMObserver_REX",
        "jitter_aware_rule_NeuralECMObserver_REX",
        "jitter_aware_rule_NeuralECMObserver_REX_TTA_voltage_only",
        "inverse_risk_NeuralECMObserver_REX",
        "inverse_risk_NeuralECMObserver_REX_TTA_voltage_only",
    ]
    rows = []
    for model in wanted:
        g = core[core["model_name"].eq(model)].dropna(subset=["omitted_MAE_pct"])
        if g.empty:
            continue
        rows.append(
            {
                "model_name": model,
                "avg_omitted_MAE_pct": float(g["omitted_MAE_pct"].mean()),
                "worst_omitted_MAE_pct": float(g["omitted_MAE_pct"].max()),
                "avg_omitted_RMSE_pct": float(g["omitted_RMSE_pct"].mean()),
                "worst_omitted_RMSE_pct": float(g["omitted_RMSE_pct"].max()),
                "avg_jitter_ratio": float(g["omitted_jitter_ratio"].mean()),
                "worst_jitter_ratio": float(g["omitted_jitter_ratio"].max()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["avg_omitted_MAE_pct", "avg_jitter_ratio"])
    out.to_csv(root / "fusion_pareto_table.csv", index=False)
    if len(out):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.scatter(out["avg_omitted_MAE_pct"], out["avg_jitter_ratio"], s=55)
        for _, r in out.iterrows():
            label = str(r["model_name"]).replace("NeuralECMObserver_", "ECM_").replace("R5_GATED_AUG_REX_best_per_fold", "R5")
            ax.annotate(label, (r["avg_omitted_MAE_pct"], r["avg_jitter_ratio"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("avg omitted MAE (%)")
        ax.set_ylabel("avg jitter ratio")
        ax.grid(True, alpha=0.25)
        fig.savefig(root / "fusion_pareto_mae_jitter.png", dpi=170)
        plt.close(fig)
    return out


def write_final_temperature_robustness_summary(root: Path) -> Path:
    pareto = pd.read_csv(root / "fusion_pareto_table.csv") if (root / "fusion_pareto_table.csv").exists() else pd.DataFrame()
    outside = pd.read_csv(root / "outside_range_fusion_focus.csv") if (root / "outside_range_fusion_focus.csv").exists() else pd.DataFrame()
    sens = pd.read_csv(root / "fusion_proxy_sensitivity.csv") if (root / "fusion_proxy_sensitivity.csv").exists() else pd.DataFrame()
    lines = ["# Final Temperature Robustness Summary\n"]
    lines.append("## 1. Representative Temperature Coverage\n")
    lines.append("With representative temperature coverage, the decomposed-feature R5/R5_GATED family remains strong for FUDS drive-cycle generalization. This should be stated as coverage-dependent robustness, not pure unseen-temperature extrapolation.\n")
    lines.append("## 2. Omitted-Temperature Failure\n")
    lines.append("When transition temperatures are omitted from training, failures appear at the omitted domain: 10C in Exp A, 0C in Exp B, and 20C in Exp C. Adding the omitted temperature stabilizes that condition, so temperature coverage sensitivity remains the central diagnosis.\n")
    lines.append("## 3. Single-Model Attempts\n")
    lines.append("REx, ShiftTau, smooth/strong-hys correctors, sequence losses, endpoint/multiscale consistency, and component sensitivity penalties did not consistently solve all omitted-temperature folds. NeuralECM reduces jitter and works well for omitted 20C and outside -10C with voltage-only TTA, but fails at omitted 0/10C and outside 50C.\n")
    lines.append("## 4. Complementary Failure Modes\n")
    lines.append("R5-based regression usually has lower MAE in some omitted folds but high local jitter. NeuralECM has much lower jitter but can have large bias/mapping errors. These are complementary, which motivates label-free fusion.\n")
    lines.append("## 5. Jitter-Aware Fusion\n")
    if len(pareto):
        lines.append(pareto.to_markdown(index=False, floatfmt=".3f"))
        lines.append("\n")
    lines.append("Jitter-aware fusion substantially reduces temporal jitter compared with R5 and previous label-free fusion. However, the best jitter-aware rows trade off worst-fold MAE, so this is temporal stabilization plus risk reduction, not a complete catastrophic-failure solution.\n")
    lines.append("## 6. Pure Extrapolation vs Voltage-Only TTA\n")
    lines.append("NeuralECM_REX_TTA and fusion rows using ECM_TTA rely on voltage-only transductive adaptation. They must be reported separately from pure extrapolation. No SOC labels are used for TTA, but the target trajectory V/I/T is used.\n")
    lines.append("## 7. Outside Range\n")
    if len(outside):
        keep = outside.dropna(subset=["omitted_MAE_pct"])[[
            "experiment",
            "model_name",
            "omitted_temperature_C",
            "omitted_MAE_pct",
            "omitted_RMSE_pct",
            "omitted_jitter_ratio",
        ]]
        lines.append(keep.to_markdown(index=False, floatfmt=".3f"))
        lines.append("\n")
    lines.append("If outside 50C remains high after R5/ECM/fusion comparison, high-temperature outside-range extrapolation remains unresolved.\n")
    lines.append("## 8. Fusion Proxy Sensitivity\n")
    if len(sens):
        lines.append(
            f"Sensitivity sweep count: {len(sens)}. Avg omitted MAE range {sens['avg_omitted_MAE_pct'].min():.2f}-{sens['avg_omitted_MAE_pct'].max():.2f}%, "
            f"avg jitter range {sens['avg_jitter_ratio'].min():.2f}-{sens['avg_jitter_ratio'].max():.2f}.\n"
        )
    lines.append("## Safe Claims\n")
    lines.append("- Jitter-aware label-free fusion substantially reduces temporal instability.\n")
    lines.append("- R5 and NeuralECM estimators exhibit complementary failure modes.\n")
    lines.append("- Voltage-only TTA improves some outside-range conditions but is not pure extrapolation.\n")
    lines.append("- Pure unseen-temperature extrapolation remains unresolved.\n")
    lines.append("## Forbidden Claims\n")
    lines.append("- The model solves pure temperature extrapolation.\n")
    lines.append("- TTA result is pure extrapolation.\n")
    lines.append("- The fusion guarantees robustness at all unseen temperatures.\n")
    lines.append("- Learned voltage components are true physical polarization/hysteresis.\n")
    path = root / "final_temperature_robustness_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def early_stop_comparison(root: Path) -> pd.DataFrame:
    hist = _read_csv(root / "neural_ecm_training_history.csv")
    rows = []
    for (exp, model), g in hist.groupby(["experiment", "model_name"]):
        g = g.sort_values("epoch").reset_index(drop=True)
        for criterion in ["loss_soc", "loss_soc_plus_voltage"]:
            if criterion == "loss_soc":
                vals = g["loss_soc"].to_numpy(np.float64)
            else:
                vals = g["loss_soc"].to_numpy(np.float64) + 0.20 * g["loss_voltage"].to_numpy(np.float64)
            best_idx = int(np.nanargmin(vals))
            rows.append(
                {
                    "experiment": exp,
                    "model_name": model,
                    "criterion": criterion,
                    "selected_epoch": int(g.loc[best_idx, "epoch"]),
                    "selected_metric": float(vals[best_idx]),
                    "final_epoch": int(g["epoch"].iloc[-1]),
                    "final_loss_soc": float(g["loss_soc"].iloc[-1]),
                    "final_loss_voltage": float(g["loss_voltage"].iloc[-1]),
                    "note": "computed from training history only; no FUDS supervised validation used",
                }
            )
        rows.append(
            {
                "experiment": exp,
                "model_name": model,
                "criterion": "LOTO_validation_score",
                "selected_epoch": np.nan,
                "selected_metric": np.nan,
                "final_epoch": int(g["epoch"].iloc[-1]),
                "final_loss_soc": float(g["loss_soc"].iloc[-1]),
                "final_loss_voltage": float(g["loss_voltage"].iloc[-1]),
                "note": "not run here because the current protocol says no train data validation split and no FUDS supervised validation",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(root / "neural_ecm_early_stop_comparison.csv", index=False)
    return out


def run_neural_ecm_diagnostics(
    cfg: CFG | None = None,
    *,
    run_prefix_tta: bool = True,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or make_cfg()
    root = _repo_root(cfg)
    outputs: dict[str, pd.DataFrame] = {}
    outputs["parameter_plausibility"] = parameter_curve_diagnostics(root)
    outputs["voltage_vs_soc_error"] = voltage_vs_soc_error_diagnostics(root)
    outputs["fusion_results"], outputs["fusion_by_temperature"], outputs["fusion_focus"] = dual_estimator_fusion(root)
    (
        outputs["jitter_aware_fusion_results"],
        outputs["jitter_aware_fusion_by_temperature"],
        outputs["jitter_aware_fusion_focus"],
    ) = jitter_aware_fusion(root)
    outputs["fusion_proxy_sensitivity"] = fusion_proxy_sensitivity(root)
    outputs["fusion_pareto"] = fusion_pareto(root)
    outputs["final_temperature_robustness_summary"] = pd.DataFrame(
        [{"path": str(write_final_temperature_robustness_summary(root))}]
    )
    outputs["early_stop_comparison"] = early_stop_comparison(root)
    if run_prefix_tta:
        pred, scales = run_tta_prefix_experiment(cfg, experiments=PREFIX_EXPERIMENTS)
        summarize_tta_prefix(root, pred, scales)
        outputs["tta_prefix_predictions"] = pred
        outputs["tta_prefix_scales"] = scales
    return outputs


if __name__ == "__main__":
    cfg = make_cfg()
    cfg.window_len = 50
    cfg.stride = 1
    cfg.ecm_chunk_stride = 1
    cfg.lstm_epochs = 300
    cfg.ecm_epochs = 300
    cfg.ecm_early_stop = True
    cfg.ecm_plateau_monitor = "loss_soc"
    cfg.ecm_plateau_warmup_epochs = 50
    cfg.ecm_plateau_patience = 35
    cfg.ecm_plateau_min_delta = 1e-4
    run_neural_ecm_diagnostics(cfg, run_prefix_tta=True)
