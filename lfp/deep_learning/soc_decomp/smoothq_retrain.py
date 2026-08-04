from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .models import DecomposedWindowDataset, build_lstm_soc_model
from .training import (
    attach_prediction_features,
    build_prediction_feature_lookup,
    make_scaled_frames_for_ablation,
    predict_loader,
)
from .variance_control import R5_GATED_FEATURES, _overall_metrics, variance_by_temperature
from .extrapolation_robustness import (
    load_filtered_feature_frames,
    train_rex_model,
    temp_key_to_c,
)
from .neural_ecm_observer import (
    ECMSpec,
    train_ecm_model,
    predict_full_trajectories,
    attach_and_focus,
)


EXPERIMENTS = {
    "Exp A": {
        "train_temps": ("N10", "0", "25", "50"),
        "omitted_temp_C": 10.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
        "note": "omitted 10C; cached corrector features reused for label-only retrain",
    },
    "Exp B": {
        "train_temps": ("N10", "10", "25", "50"),
        "omitted_temp_C": 0.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
        "note": "omitted 0C; cached corrector features reused for label-only retrain",
    },
    "Exp C": {
        "train_temps": ("N10", "0", "10", "25", "50"),
        "omitted_temp_C": 20.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
        "note": "omitted 20C; no 20C DST/US06 feature files in this cache",
    },
    "Exp D": {
        "train_temps": ("N10", "0", "10", "20", "25", "50"),
        "omitted_temp_C": 20.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_20_25_50",
        "note": "20C included; full cache with 20C DST/US06",
    },
    "Omit N10": {
        "train_temps": ("0", "10", "25", "50"),
        "omitted_temp_C": -10.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
        "note": "outside low-temperature fold",
    },
    "Omit 50": {
        "train_temps": ("N10", "0", "10", "25"),
        "omitted_temp_C": 50.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
        "note": "outside high-temperature fold",
    },
}


def clone_cfg(cfg: CFG | None = None) -> CFG:
    src = make_cfg() if cfg is None else cfg
    out = make_cfg()
    for f in fields(CFG):
        setattr(out, f.name, getattr(src, f.name))
    return out


def experiment_cfg(cfg: CFG | None, experiment: str) -> CFG:
    out = clone_cfg(cfg)
    spec = EXPERIMENTS[experiment]
    out.smoke_mode = False
    out.train_temps = spec["train_temps"]
    out.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
    out.train_drives = ("DST", "US06")
    out.eval_drive = "FUDS"
    out.use_existing_soc_cc_if_available = False
    out.use_existing_usable_if_available = False
    out.decomposed_dir = out.output_dir / spec["feature_dir"]
    return out


def configure_strict_training(cfg: CFG) -> CFG:
    cfg.window_len = 50
    cfg.stride = 1
    cfg.lstm_epochs = 300
    cfg.batch_size = 8192
    cfg.lstm_lr = 1e-3
    cfg.lstm_weight_decay = 1e-4
    cfg.lstm_print_every = 25
    cfg.lstm_early_stop = True
    cfg.lstm_plateau_warmup_epochs = 50
    cfg.lstm_plateau_patience = 35
    cfg.lstm_plateau_min_delta = 1e-4
    cfg.dataloader_num_workers = 0
    cfg.ecm_chunk_len = 50
    cfg.ecm_chunk_stride = 1
    cfg.ecm_epochs = 300
    cfg.ecm_lr = 8e-4
    cfg.ecm_early_stop = True
    cfg.ecm_plateau_monitor = "loss_soc"
    cfg.ecm_plateau_warmup_epochs = 50
    cfg.ecm_plateau_patience = 35
    cfg.ecm_plateau_min_delta = 1e-4
    return cfg


def write_final_label_policy(base_dir: Path):
    rows = [
        {
            "policy_item": "main_physical_soc_label",
            "file": "labels_physical_smoothQ.csv",
            "label_name": "physical_SOC_smoothQ",
            "use": "main physical SOC experiments",
            "notes": "Q_ref uses robust smooth capacity recommendation; cutoff is not forced to zero.",
        },
        {
            "policy_item": "sensitivity_physical_measuredQ",
            "file": "labels_physical_measuredQ.csv",
            "label_name": "physical_SOC_measuredQ",
            "use": "capacity sensitivity only",
            "notes": "-10C measuredQ is diagnostic-only because CE_ok=False.",
        },
        {
            "policy_item": "sensitivity_existing_label",
            "file": "source CSV SOC_CC / existing prediction rows",
            "label_name": "existing_label",
            "use": "backward comparison only",
            "notes": "Not selected by test error.",
        },
        {
            "policy_item": "usable_to_cutoff_label",
            "file": "labels_usable_cutoff.csv",
            "label_name": "usable_SOC_cutoff",
            "use": "remaining-to-cutoff task only",
            "notes": "Never report this as physical SOC.",
        },
        {
            "policy_item": "minus10_policy",
            "file": "labels_physical_smoothQ.csv",
            "label_name": "physical_SOC_smoothQ",
            "use": "retain with low-confidence flag",
            "notes": "label_quality=low_confidence_due_to_capacity_consistency_mismatch; Q_ref_source=smoothQ; CE_issue_flag=True.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(base_dir / "final_label_policy.csv", index=False)
    md = [
        "# Final Label Policy",
        "",
        "## Fixed Policy",
        "1. Main physical SOC label: use `labels_physical_smoothQ.csv`.",
        "2. Sensitivity labels: use `labels_physical_measuredQ.csv` and `existing_label`.",
        "3. Usable label: use `labels_usable_cutoff.csv` only for usable-to-cutoff or remaining-to-cutoff experiments.",
        "4. -10C rows are retained, not excluded.",
        "5. Usable-to-cutoff labels must not be reported as physical SOC.",
        "",
        "## -10C Rule",
        "- `label_quality = low_confidence_due_to_capacity_consistency_mismatch`.",
        "- `Q_ref_source = smoothQ` for the main physical label.",
        "- `CE_issue_flag = True`.",
        "- The CE-failed low-current capacity is not blindly used as ground-truth Q_ref.",
        "",
        "## Safe Statements",
        "- The -10C low-current capacity failed the capacity-consistency check, but its discharge capacity was close to the robust smooth Q_ref estimate.",
        "- There was no clear evidence that -10C low-current discharge capacity was underestimated by early cutoff.",
        "- -10C labels were retained with a low-confidence flag.",
        "- Physical SOC and usable-to-cutoff SOC were evaluated separately.",
        "",
        "## Forbidden Statements",
        "- -10C physical SOC was forced to zero at cutoff.",
        "- CE-failed capacity was used blindly as ground-truth Q_ref.",
        "- Usable-to-cutoff was treated as physical SOC.",
        "- Test model error was used to choose Q_ref.",
    ]
    (base_dir / "final_label_policy.md").write_text("\n".join(md), encoding="utf-8")
    return df


def load_smoothq_lookup(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "labels_physical_smoothQ.csv"
    if not path.exists():
        raise FileNotFoundError("Run capacity_audit first; labels_physical_smoothQ.csv is missing.")
    usecols = [
        "trajectory_id", "end_index", "SOC", "Q_ref_used_Ah", "label_quality",
        "Q_ref_source", "CE_ok_used", "CE_failure_reason_if_any",
    ]
    lookup = pd.read_csv(path, usecols=usecols, low_memory=False)
    lookup = lookup.rename(columns={
        "SOC": "SOC_physical_smoothQ",
        "Q_ref_used_Ah": "Q_ref_smoothQ_Ah",
    })
    lookup["CE_issue_flag"] = ~lookup["CE_ok_used"].fillna(False).astype(bool)
    lookup.loc[lookup["trajectory_id"].astype(str).str.contains("N10"), "label_quality"] = (
        "low_confidence_due_to_capacity_consistency_mismatch"
    )
    return lookup


def relabel_feature_frames(feature_frames: dict, lookup: pd.DataFrame) -> dict:
    out = {}
    keep = [
        "trajectory_id", "end_index", "SOC_physical_smoothQ", "Q_ref_smoothQ_Ah",
        "label_quality", "Q_ref_source", "CE_ok_used", "CE_issue_flag", "CE_failure_reason_if_any",
    ]
    lk = lookup[keep].drop_duplicates(["trajectory_id", "end_index"])
    for split, frames in feature_frames.items():
        out[split] = []
        for frame in frames:
            f = frame.copy()
            before = len(f)
            f = f.merge(lk, on=["trajectory_id", "end_index"], how="left", validate="one_to_one")
            if len(f) != before:
                raise AssertionError("Unexpected row count change during smoothQ relabel merge")
            if f["SOC_physical_smoothQ"].isna().any():
                tid = f["trajectory_id"].iloc[0]
                missing = int(f["SOC_physical_smoothQ"].isna().sum())
                raise ValueError(f"Missing smoothQ labels for {tid}: {missing} rows")
            f["SOC_existing_before_smoothQ"] = f["SOC_physical"]
            f["SOC_physical"] = f["SOC_physical_smoothQ"].astype(np.float32)
            f["SOC_physical_raw"] = f["SOC_physical"].astype(np.float32)
            f["Q_ref_Ah"] = f["Q_ref_smoothQ_Ah"].astype(np.float32)
            f["target_label_policy"] = "physical_smoothQ"
            out[split].append(f)
    return out


def assert_split(feature_frames: dict, experiment: str):
    train_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["train"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["test"]}
    overlap = train_ids & test_ids
    assert not overlap, f"{experiment}: train/test trajectory overlap: {sorted(overlap)}"
    train_cycles = {f["drive_cycle"].iloc[0] for f in feature_frames["train"]}
    test_cycles = {f["drive_cycle"].iloc[0] for f in feature_frames["test"]}
    assert train_cycles.issubset({"DST", "US06"}), f"{experiment}: non DST/US06 train cycle {train_cycles}"
    assert test_cycles == {"FUDS"}, f"{experiment}: expected FUDS test only, got {test_cycles}"


def load_relabelled_frames(cfg: CFG, experiment: str, lookup: pd.DataFrame) -> dict:
    frames = load_filtered_feature_frames(cfg, decomposed_dir=cfg.decomposed_dir)
    frames = relabel_feature_frames(frames, lookup)
    assert_split(frames, experiment)
    return frames


def _move(x, cfg: CFG):
    return x.to(device=device, dtype=torch.float32, non_blocking=bool(getattr(cfg, "cuda_non_blocking", True)) and device.type == "cuda")


def train_a3_smoothq(feature_frames, cfg: CFG, experiment: str):
    feature_cols = ["V_raw", "I_raw", "T"]
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    model_name = "A3_V_raw_I_T_smoothQ"
    model = build_lstm_soc_model(feature_cols, 1, cfg, "A3_V_raw_I_T").to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=bool(getattr(cfg, "dataloader_pin_memory", True)) and device.type == "cuda",
    )
    test_loader = DataLoader(test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    history = []
    best = float("inf")
    bad = 0
    warmup = int(getattr(cfg, "lstm_plateau_warmup_epochs", 50))
    patience = int(getattr(cfg, "lstm_plateau_patience", 35))
    min_delta = float(getattr(cfg, "lstm_plateau_min_delta", 1e-4))
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = _move(x, cfg)
            y = _move(y, cfg)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y, beta=0.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        row = {"experiment": experiment, "model_name": model_name, "epoch": ep, "train_loss": train_loss}
        history.append(row)
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % int(getattr(cfg, "lstm_print_every", 25)) == 0:
            print(f"{experiment} {model_name} epoch={ep} loss={train_loss:.5f}")
        if train_loss < best - min_delta:
            best = train_loss
            bad = 0
        else:
            bad += 1
        if bool(getattr(cfg, "lstm_early_stop", True)) and ep >= warmup and bad >= patience:
            history[-1]["stopped_early"] = True
            history[-1]["stop_reason"] = f"plateau monitor=train_loss best={best:.6f} patience={patience} min_delta={min_delta}"
            print(f"{experiment} {model_name} early-stop at epoch={ep}: {history[-1]['stop_reason']}")
            break
    pred = predict_loader(model, test_loader, "physical", cfg)
    pred["ablation"] = model_name
    pred["model_name"] = model_name
    return pd.DataFrame(history), pred


def attach_and_summarize(pred_rows: list[tuple[str, pd.DataFrame]], feature_frames: dict, experiment: str, omitted_temp: float):
    lookup = build_prediction_feature_lookup(feature_frames)
    attached = []
    for name, pred in pred_rows:
        p = pred.assign(split="test", ablation=name)
        attached.append(attach_prediction_features(p, lookup, ablation_name=name, target_label="physical"))
    pred = pd.concat(attached, ignore_index=True) if attached else pd.DataFrame()
    pred["experiment"] = experiment
    pred["label_policy"] = "physical_smoothQ"
    overall = _overall_metrics(pred)
    overall["experiment"] = experiment
    by_temp = variance_by_temperature(pred)
    by_temp["experiment"] = experiment
    focus_rows = []
    for model, g in by_temp.groupby("model_name"):
        omitted = g[np.isclose(g["temperature_C"].astype(float), float(omitted_temp))]
        seen = g[~np.isclose(g["temperature_C"].astype(float), float(omitted_temp))]
        focus_rows.append({
            "experiment": experiment,
            "model_name": model,
            "omitted_temperature_C": float(omitted_temp),
            "omitted_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
            "omitted_RMSE_pct": float(omitted["RMSE_pct"].iloc[0]) if len(omitted) else np.nan,
            "omitted_jitter_ratio": float(omitted["jitter_ratio"].iloc[0]) if len(omitted) else np.nan,
            "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
            "seen_RMSE_pct": float(seen["RMSE_pct"].mean()) if len(seen) else np.nan,
            "overall_MAE_pct": float(overall[overall["model_name"].eq(model)]["MAE_pct"].iloc[0]),
            "overall_RMSE_pct": float(overall[overall["model_name"].eq(model)]["RMSE_pct"].iloc[0]),
            "worst_temperature_MAE_pct": float(g["MAE_pct"].max()) if len(g) else np.nan,
            "temperature_MAE_variance": float(g["MAE_pct"].var()) if len(g) > 1 else np.nan,
            "label_policy": "physical_smoothQ",
        })
    return pred, overall, by_temp, pd.DataFrame(focus_rows)


def run_one_experiment(cfg: CFG, experiment: str, lookup: pd.DataFrame, models: tuple[str, ...]):
    ecfg = experiment_cfg(cfg, experiment)
    configure_strict_training(ecfg)
    feature_frames = load_relabelled_frames(ecfg, experiment, lookup)
    omitted = EXPERIMENTS[experiment]["omitted_temp_C"]
    pred_rows = []
    histories = []
    if "A3_V_raw_I_T" in models:
        hist, pred = train_a3_smoothq(feature_frames, ecfg, experiment)
        histories.append(hist)
        pred_rows.append(("A3_V_raw_I_T_smoothQ", pred))
    if "R5_GATED_AUG_REX" in models:
        name = "R5_GATED_AUG_REX_l1p0_smoothQ"
        _, hist, loss_by_temp, pred = train_rex_model(
            feature_frames,
            ecfg,
            name,
            lambda_rex=1.0,
            use_aug=True,
            experiment=experiment,
        )
        hist["model_name"] = name
        histories.append(hist)
        pred_rows.append((name, pred))
    if "NeuralECM_REX" in models:
        spec = ECMSpec(
            name="NeuralECMObserver_REX_smoothQ",
            use_rex=True,
            lambda_v=0.20,
            lambda_rex=0.5,
            lambda_worst=0.10,
            correction_limit=0.05,
        )
        model, hist, _ = train_ecm_model(feature_frames, ecfg, spec, experiment)
        histories.append(hist)
        pred = predict_full_trajectories(model, feature_frames["test"], spec.name)
        pred_rows.append((spec.name, pred))
    pred, overall, by_temp, focus = attach_and_summarize(pred_rows, feature_frames, experiment, omitted)
    if len(overall):
        overall["label_policy"] = "physical_smoothQ"
    if len(by_temp):
        by_temp["label_policy"] = "physical_smoothQ"
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    history["experiment"] = experiment
    history["label_policy"] = "physical_smoothQ"
    return pred, overall, by_temp, focus, history


def summarize_focus(focus: pd.DataFrame) -> pd.DataFrame:
    if focus.empty:
        return pd.DataFrame()
    rows = []
    omitted = focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])]
    for model, g in omitted.groupby("model_name"):
        rows.append({
            "model_name": model,
            "scope": "ExpA_B_C_omitted",
            "average_omitted_MAE_pct": float(g["omitted_MAE_pct"].mean()),
            "worst_omitted_MAE_pct": float(g["omitted_MAE_pct"].max()),
            "average_omitted_RMSE_pct": float(g["omitted_RMSE_pct"].mean()),
            "average_omitted_jitter_ratio": float(g["omitted_jitter_ratio"].mean()),
            "n_folds": int(g["experiment"].nunique()),
        })
    outside = focus[focus["experiment"].isin(["Omit N10", "Omit 50"])]
    for model, g in outside.groupby("model_name"):
        rows.append({
            "model_name": model,
            "scope": "outside_range",
            "average_omitted_MAE_pct": float(g["omitted_MAE_pct"].mean()),
            "worst_omitted_MAE_pct": float(g["omitted_MAE_pct"].max()),
            "average_omitted_RMSE_pct": float(g["omitted_RMSE_pct"].mean()),
            "average_omitted_jitter_ratio": float(g["omitted_jitter_ratio"].mean()),
            "n_folds": int(g["experiment"].nunique()),
        })
    return pd.DataFrame(rows)


def relabel_reference_rows(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "label_sensitivity_by_temperature.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["label_variant"].eq("physical_smoothQ")].copy()
    rows = []
    if "experiment" not in df.columns:
        df["experiment"] = np.nan
    for _, r in df.iterrows():
        rows.append({
            "experiment": r.get("experiment", np.nan),
            "model_name": r["model_name"],
            "temperature_C": r.get("temperature_C", np.nan),
            "reference_MAE_pct": r["MAE_pct"],
            "reference_RMSE_pct": r["RMSE_pct"],
            "reference_source_file": r["model_source_file"],
            "reference_mode": r["sensitivity_mode"],
            "label_variant": "physical_smoothQ",
        })
    return pd.DataFrame(rows)


def compare_retraining_effect(base_dir: Path, retrain_results: pd.DataFrame, retrain_by_temp: pd.DataFrame, retrain_focus: pd.DataFrame):
    reference = relabel_reference_rows(base_dir)
    comp_rows = []
    # Prefer fold-level focus where available.
    if len(reference):
        for _, r in retrain_focus.iterrows():
            exp = r["experiment"]
            model = r["model_name"]
            candidates = reference.copy()
            if pd.notna(exp):
                candidates = candidates[candidates["experiment"].astype(str).eq(str(exp))]
            token_map = {
                "A3_V_raw_I_T_smoothQ": "A3_V_raw_I_T",
                "R5_GATED_AUG_REX_l1p0_smoothQ": "R5_GATED_AUG_REX_l1p0",
                "NeuralECMObserver_REX_smoothQ": "NeuralECMObserver_REX",
            }
            token = token_map.get(model, model.replace("_smoothQ", ""))
            candidates = candidates[candidates["model_name"].astype(str).eq(token)]
            candidates = candidates[np.isclose(candidates["temperature_C"].astype(float), float(r["omitted_temperature_C"]))]
            ref = candidates.iloc[0] if len(candidates) else None
            comp_rows.append({
                "experiment": exp,
                "model_name": model,
                "omitted_temperature_C": r["omitted_temperature_C"],
                "smoothQ_retrain_omitted_MAE_pct": r["omitted_MAE_pct"],
                "prediction_relabel_reference_MAE_pct": float(ref["reference_MAE_pct"]) if ref is not None else np.nan,
                "delta_retrain_minus_relabel_MAE_pct": r["omitted_MAE_pct"] - float(ref["reference_MAE_pct"]) if ref is not None else np.nan,
                "minus10_change_ge_0p5pct_flag": bool(
                    np.isclose(float(r["omitted_temperature_C"]), -10.0)
                    and ref is not None
                    and abs(r["omitted_MAE_pct"] - float(ref["reference_MAE_pct"])) >= 0.5
                ),
                "reference_mode": ref["reference_mode"] if ref is not None else "not_available",
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(base_dir / "label_retraining_effect_summary.csv", index=False)
    lines = [
        "# Label Retraining Effect Report",
        "",
        "SmoothQ retraining used `labels_physical_smoothQ.csv` as the main physical SOC target.",
        "The comparison against prediction relabel diagnostics should be interpreted carefully: prediction relabeling changes y_true only, while retraining changes fitted model weights.",
        "",
    ]
    if len(comp):
        lines.extend([
            "## Fold-Level Comparison",
            comp.to_markdown(index=False),
            "",
        ])
        big = comp[np.abs(comp["delta_retrain_minus_relabel_MAE_pct"]) >= 0.5]
        if len(big):
            lines.append("At least one fold differs by >= 0.5 percentage points; that fold should be reinterpreted under the smoothQ policy.")
        else:
            lines.append("No available fold comparison changed by >= 0.5 percentage points; label policy does not appear to overturn the main conclusions in this minimal check.")
    else:
        lines.append("No matching prediction relabel reference rows were available.")
    lines.extend([
        "",
        "## Guardrails",
        "- This run does not use FUDS for supervised training.",
        "- Train/test split is file-level by trajectory_id.",
        "- Scalers are fit only on train feature frames inside each model training call.",
        "- Q_ref was not selected using test model error.",
    ])
    (base_dir / "label_retraining_effect_report.md").write_text("\n".join(lines), encoding="utf-8")
    return comp


def write_final_summary(base_dir: Path, focus: pd.DataFrame, aggregate: pd.DataFrame):
    lines = [
        "# Final Capacity Label Summary",
        "",
        "## Label Policy",
        "- Main physical SOC: `labels_physical_smoothQ.csv`.",
        "- Sensitivity labels: `labels_physical_measuredQ.csv` and existing labels.",
        "- Usable-to-cutoff: `labels_usable_cutoff.csv`, not physical SOC.",
        "- -10C labels are retained with `low_confidence_due_to_capacity_consistency_mismatch`.",
        "",
        "## Safe Interpretation",
        "- The -10C low-current capacity failed the capacity-consistency check, but its discharge capacity was close to the robust smooth Q_ref estimate.",
        "- There was no clear evidence that -10C low-current discharge capacity was underestimated by early cutoff.",
        "- -10C labels were retained with a low-confidence flag.",
        "- Physical SOC and usable-to-cutoff SOC were evaluated separately.",
        "",
        "## Forbidden Interpretation",
        "- -10C physical SOC was forced to zero at cutoff.",
        "- CE-failed capacity was used blindly as ground-truth Q_ref.",
        "- Usable-to-cutoff was treated as physical SOC.",
        "- Test model error was used to choose Q_ref.",
        "",
        "## SmoothQ Retrain Aggregate",
        aggregate.to_markdown(index=False) if len(aggregate) else "SmoothQ retrain aggregate is not available.",
        "",
        "## Retrain Scope Caveat",
        "This is a minimal label-policy retraining check. SOC estimators were retrained with train-only feature scaling and file-level DST/US06 -> FUDS splits. "
        "The voltage-decomposition corrector features were reused from cached feature folders because the label audit changes SOC/Q_ref targets, not the unsupervised voltage decomposition loss. "
        "Rows in `smoothQ_retrain_metadata.csv` explicitly mark `corrector_feature_cache_reused=True`; do not present this artifact as a fresh corrector-refit experiment.",
        "",
        "## SmoothQ Omitted Focus",
        focus.to_markdown(index=False) if len(focus) else "SmoothQ focus table is not available.",
    ]
    (base_dir / "final_capacity_label_summary.md").write_text("\n".join(lines), encoding="utf-8")
    # Append the same policy note to the capacity audit report for convenience.
    cap = base_dir / "capacity_audit_report.md"
    if cap.exists():
        text = cap.read_text(encoding="utf-8")
        marker = "\n\n## Final Label Policy Update\n"
        update = marker + "\n".join(lines[1:18])
        if "## Final Label Policy Update" not in text:
            cap.write_text(text + update, encoding="utf-8")


def run_smoothq_retrain(
    cfg: CFG | None = None,
    *,
    experiments: tuple[str, ...] = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50"),
    models: tuple[str, ...] = ("A3_V_raw_I_T", "R5_GATED_AUG_REX", "NeuralECM_REX"),
):
    configure_torch_runtime()
    base_cfg = configure_strict_training(clone_cfg(cfg))
    base_dir = base_cfg.output_dir
    write_final_label_policy(base_dir)
    lookup = load_smoothq_lookup(base_dir)
    all_pred, all_results, all_by_temp, all_focus, all_history = [], [], [], [], []
    meta_rows = []
    for experiment in experiments:
        print(f"=== smoothQ retrain {experiment} ===")
        pred, overall, by_temp, focus, history = run_one_experiment(base_cfg, experiment, lookup, models)
        spec = EXPERIMENTS[experiment]
        for model in focus["model_name"].unique() if len(focus) else []:
            meta_rows.append({
                "experiment": experiment,
                "model_name": model,
                "train_temps": ",".join(spec["train_temps"]),
                "test_cycle": "FUDS",
                "omitted_temperature_C": spec["omitted_temp_C"],
                "feature_dir": spec["feature_dir"],
                "note": spec["note"],
                "corrector_feature_cache_reused": True,
                "supervised_fuds_training_used": False,
                "q_ref_selected_using_test_error": False,
            })
        all_pred.append(pred)
        all_results.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
        all_history.append(history)
        # Save after every fold so an interrupted run keeps partial artifacts.
        pd.concat(all_pred, ignore_index=True).to_csv(base_dir / "smoothQ_retrain_prediction_rows.csv", index=False)
        pd.concat(all_results, ignore_index=True).to_csv(base_dir / "smoothQ_retrain_results.csv", index=False)
        pd.concat(all_by_temp, ignore_index=True).to_csv(base_dir / "smoothQ_retrain_by_temperature.csv", index=False)
        pd.concat(all_focus, ignore_index=True).to_csv(base_dir / "smoothQ_retrain_omitted_focus.csv", index=False)
        pd.concat(all_history, ignore_index=True).to_csv(base_dir / "smoothQ_retrain_history.csv", index=False)
        pd.DataFrame(meta_rows).to_csv(base_dir / "smoothQ_retrain_metadata.csv", index=False)
    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    aggregate = summarize_focus(focus)
    aggregate.to_csv(base_dir / "smoothQ_vs_existing_label_comparison.csv", index=False)
    compare_retraining_effect(base_dir, results, by_temp, focus)
    write_final_summary(base_dir, focus, aggregate)
    return {
        "predictions": pred_all,
        "results": results,
        "by_temperature": by_temp,
        "focus": focus,
        "aggregate": aggregate,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrain minimal SOC models using capacity-audited smoothQ labels.")
    parser.add_argument("--experiments", default="Exp A,Exp B,Exp C,Exp D,Omit N10,Omit 50")
    parser.add_argument("--models", default="A3_V_raw_I_T,R5_GATED_AUG_REX,NeuralECM_REX")
    args = parser.parse_args()
    exps = tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    out = run_smoothq_retrain(experiments=exps, models=models)
    print("smoothQ retrain complete")
    print(out["focus"].to_string(index=False))


if __name__ == "__main__":
    main()
