from dataclasses import fields
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import load_and_prepare_data
from .models import DecomposedWindowDataset, build_corrector, build_lstm_soc_model, collate_meta_to_frame
from .corrector import run_corrector_pretraining
from .features import extract_all_feature_frames
from .training import (
    ABLATIONS,
    attach_prediction_features,
    build_prediction_feature_lookup,
    make_scaled_frames_for_ablation,
)
from .variance_control import (
    R5_GATED_FEATURES,
    augment_components,
    load_feature_frame_dict_from_csv,
    variance_by_temperature,
    _overall_metrics,
)
from .ood_diagnostic import compute_ood_feature_distance_scores

try:
    from IPython.display import display
except Exception:
    display = print


EXP_SPECS = {
    "Exp A": {
        "train_temps": ("N10", "0", "25", "50"),
        "omitted_temp_C": 10.0,
        "feature_dir": "decomposed_features",
    },
    "Exp B": {
        "train_temps": ("N10", "10", "25", "50"),
        "omitted_temp_C": 0.0,
        "feature_dir": "decomposed_features",
    },
    "Exp C": {
        "train_temps": ("N10", "0", "10", "25", "50"),
        "omitted_temp_C": 20.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
    },
}

REX_LAMBDAS = (0.1, 0.5, 1.0, 2.0)


def clone_cfg(cfg: CFG | None = None) -> CFG:
    src = make_cfg() if cfg is None else cfg
    out = make_cfg()
    for f in fields(CFG):
        setattr(out, f.name, getattr(src, f.name))
    return out


def temp_key_to_c(v) -> float:
    s = str(v).strip().upper()
    return -float(s[1:]) if s.startswith("N") else float(s)


def exp_cfg(cfg: CFG | None, experiment: str, feature_dir: str | None = None) -> CFG:
    out = clone_cfg(cfg)
    spec = EXP_SPECS[experiment]
    out.smoke_mode = False
    out.use_existing_soc_cc_if_available = False
    out.use_existing_usable_if_available = False
    out.train_temps = spec["train_temps"]
    out.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
    out.train_drives = ("DST", "US06")
    out.eval_drive = "FUDS"
    out.decomposed_dir = out.output_dir / (feature_dir or spec["feature_dir"])
    out.decomposed_dir.mkdir(parents=True, exist_ok=True)
    return out


def load_filtered_feature_frames(cfg: CFG, decomposed_dir=None):
    frames = load_feature_frame_dict_from_csv(cfg, decomposed_dir=decomposed_dir or cfg.decomposed_dir)
    train_temps = {temp_key_to_c(t) for t in cfg.train_temps}
    eval_temps = {temp_key_to_c(t) for t in cfg.eval_temps}
    out = {"train": [], "valid": [], "test": []}
    for split, split_frames in frames.items():
        for frame in split_frames:
            temp = float(frame["temperature"].iloc[0])
            if split == "train" and temp not in train_temps:
                continue
            if split == "test" and temp not in eval_temps:
                continue
            out[split].append(frame)
    train_ids = {f["trajectory_id"].iloc[0] for f in out["train"]}
    test_ids = {f["trajectory_id"].iloc[0] for f in out["test"]}
    assert train_ids.isdisjoint(test_ids), "Train/test leakage: same trajectory_id in both splits"
    if not out["train"] or not out["test"]:
        raise ValueError(f"Empty train/test split after filtering {cfg.decomposed_dir}")
    return out


def dataset_endpoint_temperatures(ds: DecomposedWindowDataset):
    temps = []
    for fi, _, end in ds.index:
        temps.append(float(ds.frames[fi]["temperature"][end]))
    return np.asarray(temps, dtype=np.float32)


def temperature_balanced_loader(ds, cfg: CFG, *, shuffle=True):
    num_workers = int(getattr(cfg, "dataloader_num_workers", 0))
    loader_kwargs = {
        "batch_size": int(cfg.batch_size),
        "num_workers": num_workers,
        "pin_memory": bool(getattr(cfg, "dataloader_pin_memory", True)) and device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(getattr(cfg, "dataloader_prefetch_factor", 4))
    if not shuffle:
        return DataLoader(ds, shuffle=False, **loader_kwargs)
    temps = dataset_endpoint_temperatures(ds)
    unique, counts = np.unique(temps, return_counts=True)
    count_map = {float(t): int(c) for t, c in zip(unique, counts)}
    weights = np.asarray([1.0 / count_map[float(t)] for t in temps], dtype=np.float64)
    generator = None
    sampler_seed = getattr(cfg, "sampler_seed", None)
    if sampler_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(sampler_seed))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    return DataLoader(
        ds,
        sampler=sampler,
        **loader_kwargs,
    )


def write_sampler_distribution(ds, cfg: CFG, experiment: str):
    temps = dataset_endpoint_temperatures(ds)
    rows = []
    for temp, count in zip(*np.unique(temps, return_counts=True)):
        rows.append({
            "experiment": experiment,
            "temperature_C": float(temp),
            "n_train_windows_before_balancing": int(count),
            "sampling_weight": float(1.0 / count),
        })
    out = pd.DataFrame(rows)
    path = cfg.output_dir / "sampler_temperature_distribution.csv"
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, out], ignore_index=True).drop_duplicates()
    out.to_csv(path, index=False)
    return out


def _move(x, cfg: CFG):
    return x.to(device=device, dtype=torch.float32, non_blocking=bool(getattr(cfg, "cuda_non_blocking", True)) and device.type == "cuda")


def rex_batch_loss(model, x, y, meta, lambda_rex=0.5, aug_spec=None):
    pred = model(x)
    temps = meta["temperature"]
    if torch.is_tensor(temps):
        temps_t = temps.to(device=pred.device, dtype=torch.float32)
    else:
        temps_t = torch.as_tensor(temps, device=pred.device, dtype=torch.float32)
    losses = []
    loss_by_temp = {}
    for temp in torch.unique(temps_t):
        mask = temps_t == temp
        if mask.any():
            lt = F.l1_loss(pred[mask], y[mask])
            losses.append(lt)
            loss_by_temp[float(temp.detach().cpu())] = lt
    if not losses:
        l_mean = F.l1_loss(pred, y)
        l_rex = pred.new_tensor(0.0)
    else:
        stack = torch.stack(losses)
        l_mean = stack.mean()
        l_rex = stack.var(unbiased=False) if len(losses) > 1 else stack.new_tensor(0.0)
    total = l_mean + float(lambda_rex) * l_rex
    if aug_spec is not None and float(aug_spec.get("lambda_aug", 0.0)) > 0:
        x_aug = augment_components(x, aug_spec["features"], aug_spec)
        pred_aug = model(x_aug)
        total = total + float(aug_spec.get("lambda_aug", 0.0)) * F.smooth_l1_loss(pred_aug, pred.detach(), beta=0.01)
    return total, l_mean.detach(), l_rex.detach(), loss_by_temp


@torch.no_grad()
def predict_model(model, loader, cfg: CFG):
    rows = []
    model.eval()
    for x, y, meta in loader:
        yp = model(_move(x, cfg)).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = "physical"
        mdf["y_true"] = yy
        mdf["y_pred"] = yp
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def train_rex_model(feature_frames, cfg: CFG, model_name, lambda_rex=0.5, *, use_aug=False, experiment="Exp"):
    feature_cols = R5_GATED_FEATURES
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    if len(train_ds) == 0:
        raise ValueError("No train windows for REx model")
    write_sampler_distribution(train_ds, cfg, experiment)
    train_loader = temperature_balanced_loader(train_ds, cfg, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    model = build_lstm_soc_model(feature_cols, 1, cfg, "R5_GATED").to(device)
    aug_spec = None
    if use_aug:
        aug_spec = {
            "features": feature_cols,
            "lambda_aug": 0.05,
            "component_noise_std": 0.01,
            "component_dropout_p": 0.10,
        }
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    history = []
    temp_loss_rows = []
    early_stop = bool(getattr(cfg, "lstm_early_stop", True))
    warmup_epochs = int(getattr(cfg, "lstm_plateau_warmup_epochs", 50))
    patience = int(getattr(cfg, "lstm_plateau_patience", 35))
    min_delta = float(getattr(cfg, "lstm_plateau_min_delta", 1e-4))
    best_metric = float("inf")
    bad_epochs = 0
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        mean_losses = []
        rex_losses = []
        temp_meters = {}
        for x, y, meta in train_loader:
            x = _move(x, cfg)
            y = _move(y, cfg)
            loss, l_mean, l_rex, by_temp = rex_batch_loss(model, x, y, meta, lambda_rex=lambda_rex, aug_spec=aug_spec)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mean_losses.append(float(l_mean.cpu()))
            rex_losses.append(float(l_rex.cpu()))
            for temp, lt in by_temp.items():
                temp_meters.setdefault(temp, []).append(float(lt.detach().cpu()))
        row = {
            "experiment": experiment,
            "model_name": model_name,
            "epoch": ep,
            "lambda_rex": float(lambda_rex),
            "train_loss": float(np.mean(losses)),
            "train_mean_domain_loss": float(np.mean(mean_losses)),
            "train_rex_variance": float(np.mean(rex_losses)),
        }
        history.append(row)
        for temp, vals in temp_meters.items():
            temp_loss_rows.append({
                "experiment": experiment,
                "model_name": model_name,
                "lambda_rex": float(lambda_rex),
                "epoch": ep,
                "temperature_C": float(temp),
                "train_loss_mean": float(np.mean(vals)),
            })
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 25)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"{experiment} {model_name} epoch={ep} loss={row['train_loss']:.5f} rex={row['train_rex_variance']:.6f}")
        metric = float(row["train_loss"])
        if metric < best_metric - min_delta:
            best_metric = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
        if early_stop and ep >= warmup_epochs and bad_epochs >= patience:
            row["stopped_early"] = True
            row["stop_reason"] = (
                f"plateau monitor=train_loss best={best_metric:.6f} "
                f"patience={patience} min_delta={min_delta}"
            )
            history[-1] = row
            print(f"{experiment} {model_name} early-stop at epoch={ep}: {row['stop_reason']}")
            break
    pred = predict_model(model, test_loader, cfg)
    return model, pd.DataFrame(history), pd.DataFrame(temp_loss_rows), pred


def train_standard_gated(feature_frames, cfg: CFG, model_name, *, use_aug=False, experiment="Exp"):
    feature_cols = R5_GATED_FEATURES
    scaled, _ = make_scaled_frames_for_ablation(feature_frames, feature_cols)
    train_ds = DecomposedWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    train_loader = temperature_balanced_loader(train_ds, cfg, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    model = build_lstm_soc_model(feature_cols, 1, cfg, "R5_GATED").to(device)
    aug_spec = None
    if use_aug:
        aug_spec = {
            "features": feature_cols,
            "lambda_aug": 0.05,
            "component_noise_std": 0.01,
            "component_dropout_p": 0.10,
        }
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    history = []
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = _move(x, cfg)
            y = _move(y, cfg)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y, beta=0.02)
            if aug_spec is not None:
                x_aug = augment_components(x, feature_cols, aug_spec)
                loss = loss + aug_spec["lambda_aug"] * F.smooth_l1_loss(model(x_aug), pred.detach(), beta=0.01)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"experiment": experiment, "model_name": model_name, "epoch": ep, "train_loss": float(np.mean(losses))}
        history.append(row)
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 25)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"{experiment} {model_name} epoch={ep} loss={row['train_loss']:.5f}")
    pred = predict_model(model, test_loader, cfg)
    return model, pd.DataFrame(history), pd.DataFrame(), pred


def attach_and_summarize(pred_rows, feature_frames, cfg: CFG, experiment: str, omitted_temp):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    attached = []
    for name, pred in pred_rows:
        p = pred.assign(split="test", ablation=name)
        attached.append(attach_prediction_features(p, feature_lookup, ablation_name=name, target_label="physical"))
    pred = pd.concat(attached, ignore_index=True) if attached else pd.DataFrame()
    pred["experiment"] = experiment
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
        })
    focus = pd.DataFrame(focus_rows)
    return pred, overall, by_temp, focus


def run_rex_experiment(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C"),
    lambda_rex_values=REX_LAMBDAS,
    include_baselines=True,
):
    configure_torch_runtime()
    cfg = cfg or make_cfg()
    all_pred, all_results, all_by_temp, all_focus, all_loss = [], [], [], [], []
    for experiment in experiments:
        ecfg = exp_cfg(cfg, experiment)
        feature_frames = load_filtered_feature_frames(ecfg)
        omitted = EXP_SPECS[experiment]["omitted_temp_C"]
        pred_rows = []
        if include_baselines:
            for name, use_aug in [("R5_GATED", False), ("R5_GATED_AUG", True)]:
                _, hist, _, pred = train_standard_gated(feature_frames, ecfg, name, use_aug=use_aug, experiment=experiment)
                pred_rows.append((name, pred))
        for lam in lambda_rex_values:
            for base, use_aug in [("R5_GATED_REX", False), ("R5_GATED_AUG_REX", True)]:
                name = f"{base}_l{str(lam).replace('.', 'p')}"
                _, hist, loss_by_temp, pred = train_rex_model(
                    feature_frames,
                    ecfg,
                    name,
                    lambda_rex=lam,
                    use_aug=use_aug,
                    experiment=experiment,
                )
                pred_rows.append((name, pred))
                all_loss.append(loss_by_temp)
        pred, overall, by_temp, focus = attach_and_summarize(pred_rows, feature_frames, ecfg, experiment, omitted)
        all_pred.append(pred)
        all_results.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
    pred_all = pd.concat(all_pred, ignore_index=True)
    results = pd.concat(all_results, ignore_index=True)
    by_temp = pd.concat(all_by_temp, ignore_index=True)
    focus = pd.concat(all_focus, ignore_index=True)
    loss_by_temp = pd.concat(all_loss, ignore_index=True) if all_loss else pd.DataFrame()
    pred_all.to_csv(cfg.output_dir / "rex_prediction_rows.csv", index=False)
    results.to_csv(cfg.output_dir / "rex_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "rex_by_temperature.csv", index=False)
    focus.to_csv(cfg.output_dir / "rex_omitted_temp_focus.csv", index=False)
    loss_by_temp.to_csv(cfg.output_dir / "rex_loss_by_temperature.csv", index=False)
    print("REx omitted-temperature focus:")
    display(focus.sort_values(["experiment", "omitted_MAE_pct"]).head(30))
    return {"predictions": pred_all, "results": results, "by_temperature": by_temp, "focus": focus, "loss_by_temperature": loss_by_temp}


def run_loto_validation(cfg: CFG | None = None, *, lambda_rex_values=REX_LAMBDAS):
    cfg = cfg or make_cfg()
    rex = run_rex_experiment(cfg, experiments=("Exp A", "Exp B", "Exp C"), lambda_rex_values=lambda_rex_values, include_baselines=True)
    focus = rex["focus"].copy()
    focus.to_csv(cfg.output_dir / "loto_validation_results.csv", index=False)
    summary = (
        focus.groupby("model_name")
        .agg(
            meta_score_MAE=("omitted_MAE_pct", "mean"),
            meta_score_RMSE_plus_0p1_jitter=("omitted_RMSE_pct", lambda s: float(s.mean())),
            worst_omitted_MAE_pct=("omitted_MAE_pct", "max"),
            seen_MAE_pct=("seen_MAE_pct", "mean"),
            n_folds=("experiment", "nunique"),
        )
        .reset_index()
        .sort_values("meta_score_MAE")
    )
    jitter = focus.groupby("model_name")["omitted_jitter_ratio"].mean().reindex(summary["model_name"]).to_numpy()
    summary["meta_score_RMSE_plus_0p1_jitter"] = (
        focus.groupby("model_name")["omitted_RMSE_pct"].mean().reindex(summary["model_name"]).to_numpy()
        + 0.1 * jitter
    )
    summary.to_csv(cfg.output_dir / "loto_model_selection_summary.csv", index=False)
    return {"validation": focus, "selection": summary, "rex": rex}


def make_shift_tau_cfg(cfg: CFG | None, experiment: str, variant: str) -> CFG:
    out = exp_cfg(cfg, experiment, feature_dir=f"decomposed_features_{experiment.replace(' ', '_').lower()}_{variant}")
    out.corrector_variant = variant
    out.corrector_epochs = min(int(getattr(out, "corrector_epochs", 10)), 10)
    out.corrector_train_segment_len = int(getattr(out, "corrector_train_segment_len", 4096) or 4096)
    out.corrector_segments_per_profile_per_epoch = int(getattr(out, "corrector_segments_per_profile_per_epoch", 4) or 4)
    out.corrector_profile_batch_size = int(getattr(out, "corrector_profile_batch_size", 8) or 8)
    out.lambda_temp_tau_reg = float(getattr(out, "lambda_temp_tau_reg", 0.002) or 0.002)
    out.lambda_R0_smooth = max(float(getattr(out, "lambda_R0_smooth", 0.0) or 0.0), 0.02)
    return out


def train_shift_tau_corrector_features(cfg: CFG | None = None, *, experiment="Exp C", variant="shift_tau_arrhenius"):
    scfg = make_shift_tau_cfg(cfg, experiment, variant)
    configure_torch_runtime()
    data = load_and_prepare_data(scfg)
    corrector = build_corrector(scfg, device)
    history = run_corrector_pretraining(corrector, data["train_profiles"], scfg, data["v_scaler"])
    feature_frames = extract_all_feature_frames(
        corrector,
        data["train_profiles"],
        data["valid_profiles"],
        data["test_profiles"],
        scfg,
        data["v_scaler"],
    )
    history.to_csv(scfg.output_dir / f"{variant}_{experiment.replace(' ', '_')}_corrector_history.csv", index=False)
    return {"cfg": scfg, "feature_frames": feature_frames, "history": history}


def component_summary(feature_frames):
    rows = []
    for split, frames in feature_frames.items():
        for frame in frames:
            row = {
                "split": split,
                "trajectory_id": frame["trajectory_id"].iloc[0],
                "temperature_C": float(frame["temperature"].iloc[0]),
                "drive_cycle": frame["drive_cycle"].iloc[0],
            }
            for c in ["V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]:
                x = frame[c].to_numpy(float)
                row[f"{c}_mean"] = float(np.nanmean(x))
                row[f"{c}_std"] = float(np.nanstd(x))
                row[f"{c}_hf"] = float(np.nanmean(np.diff(x, n=2) ** 2)) if len(x) > 2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def run_shift_tau_soc_experiment(
    cfg: CFG | None = None,
    *,
    experiments=("Exp C",),
    variants=("shift_tau_arrhenius", "shift_tau_mlp_bounded", "shift_tau_hybrid"),
    lambda_rex=0.5,
):
    all_summary, all_by_temp, all_focus, all_component = [], [], [], []
    for experiment in experiments:
        for variant in variants:
            trained = train_shift_tau_corrector_features(cfg, experiment=experiment, variant=variant)
            scfg = trained["cfg"]
            feature_frames = load_filtered_feature_frames(scfg, decomposed_dir=scfg.decomposed_dir)
            comp = component_summary(feature_frames)
            comp["experiment"] = experiment
            comp["variant"] = variant
            all_component.append(comp)
            _, _, loss_by_temp, pred = train_rex_model(
                feature_frames,
                scfg,
                f"R5_GATED_REX_{variant}",
                lambda_rex=lambda_rex,
                use_aug=False,
                experiment=experiment,
            )
            attached, overall, by_temp, focus = attach_and_summarize(
                [(f"R5_GATED_REX_{variant}", pred)],
                feature_frames,
                scfg,
                experiment,
                EXP_SPECS[experiment]["omitted_temp_C"],
            )
            overall["variant"] = variant
            by_temp["variant"] = variant
            focus["variant"] = variant
            all_summary.append(overall)
            all_by_temp.append(by_temp)
            all_focus.append(focus)
    comp = pd.concat(all_component, ignore_index=True) if all_component else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    comp.to_csv((cfg or make_cfg()).output_dir / "shift_tau_corrector_component_summary.csv", index=False)
    summary.to_csv((cfg or make_cfg()).output_dir / "shift_tau_soc_results.csv", index=False)
    by_temp.to_csv((cfg or make_cfg()).output_dir / "shift_tau_by_temperature.csv", index=False)
    focus.to_csv((cfg or make_cfg()).output_dir / "shift_tau_omitted_temp_focus.csv", index=False)
    feature_distance = compute_shift_tau_feature_distance_summary(cfg or make_cfg(), experiments=experiments, variants=variants)
    return {"component_summary": comp, "results": summary, "by_temperature": by_temp, "focus": focus}


def _s3_distance_by_temperature(feature_frames):
    cols = ["V_raw", "I_raw", "T", "V_corr_raw", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]
    train = pd.concat([f[cols + ["temperature"]] for f in feature_frames["train"]], ignore_index=True)
    test = pd.concat([f[cols + ["temperature"]] for f in feature_frames["test"]], ignore_index=True)
    mu = train[cols].mean()
    sd = train[cols].std().replace(0.0, 1.0)
    train_z = ((train[cols] - mu) / sd).to_numpy(np.float64)
    test_z = ((test[cols] - mu) / sd).to_numpy(np.float64)
    cov = np.cov(train_z, rowvar=False)
    reg = 1e-3 * np.trace(cov) / max(1, cov.shape[0])
    inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * max(reg, 1e-6))
    centroids = []
    for temp, g in train.assign(_row=np.arange(len(train))).groupby("temperature"):
        centroids.append((float(temp), train_z[g["_row"].to_numpy(int)].mean(axis=0)))
    c_stack = np.stack([c for _, c in centroids], axis=0)
    d = np.sqrt(((test_z[:, None, :] - c_stack[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    delta = test_z - train_z.mean(axis=0)
    maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv, delta), 0.0))
    out = test[["temperature"]].copy()
    out["distance_S3_centroid"] = d
    out["mahalanobis_S3"] = maha
    return (
        out.groupby("temperature")[["distance_S3_centroid", "mahalanobis_S3"]]
        .mean()
        .reset_index()
        .rename(columns={"temperature": "temperature_C"})
    )


def compute_shift_tau_feature_distance_summary(cfg: CFG | None = None, *, experiments=("Exp C",), variants=("shift_tau_arrhenius", "shift_tau_mlp_bounded", "shift_tau_hybrid")):
    cfg = cfg or make_cfg()
    rows = []
    for experiment in experiments:
        for variant in variants:
            scfg = make_shift_tau_cfg(cfg, experiment, variant)
            if not scfg.decomposed_dir.exists():
                continue
            try:
                feature_frames = load_filtered_feature_frames(scfg, decomposed_dir=scfg.decomposed_dir)
                dist = _s3_distance_by_temperature(feature_frames)
                dist["experiment"] = experiment
                dist["variant"] = variant
                rows.append(dist)
            except Exception as exc:
                warnings.warn(f"Could not compute shift-tau feature distance for {experiment}/{variant}: {exc}")
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(cfg.output_dir / "shift_tau_corrector_feature_distance.csv", index=False)
    return out


def write_rex_shift_tau_ood_comparison(cfg: CFG | None = None):
    cfg = cfg or make_cfg()
    rows = []
    for path in sorted(cfg.output_dir.glob("*ood_distance_by_temperature.csv")):
        if path.name.startswith("shift_tau_"):
            continue
        df = pd.read_csv(path)
        if "temperature_C" not in df:
            continue
        for _, r in df.iterrows():
            if float(r["temperature_C"]) in {0.0, 10.0, 20.0}:
                rows.append({
                    "source": path.name,
                    "temperature_C": float(r["temperature_C"]),
                    "mahalanobis_S3": r.get("mahalanobis_S3", np.nan),
                    "distance_S3_centroid": r.get("distance_S3_centroid", np.nan),
                    "abs_error": r.get("abs_error", np.nan),
                })
    shift_dist = pd.read_csv(cfg.output_dir / "shift_tau_corrector_feature_distance.csv") if (cfg.output_dir / "shift_tau_corrector_feature_distance.csv").exists() else pd.DataFrame()
    shift_temp = pd.read_csv(cfg.output_dir / "shift_tau_by_temperature.csv") if (cfg.output_dir / "shift_tau_by_temperature.csv").exists() else pd.DataFrame()
    if len(shift_dist):
        for _, r in shift_dist.iterrows():
            if float(r["temperature_C"]) not in {0.0, 10.0, 20.0}:
                continue
            err = np.nan
            if len(shift_temp):
                m = shift_temp[
                    (shift_temp.get("variant", "").eq(r.get("variant")))
                    & np.isclose(shift_temp["temperature_C"].astype(float), float(r["temperature_C"]))
                ]
                if len(m):
                    err = float(m["MAE"].iloc[0])
            rows.append({
                "source": f"{r.get('variant')}_{r.get('experiment')}_shift_tau_features",
                "temperature_C": float(r["temperature_C"]),
                "mahalanobis_S3": r.get("mahalanobis_S3", np.nan),
                "distance_S3_centroid": r.get("distance_S3_centroid", np.nan),
                "abs_error": err,
            })
    out = pd.DataFrame(rows)
    out.to_csv(cfg.output_dir / "rex_shift_tau_ood_comparison.csv", index=False)
    return out


def write_extrapolation_robustness_report(cfg: CFG | None = None):
    cfg = cfg or make_cfg()
    focus = pd.read_csv(cfg.output_dir / "rex_omitted_temp_focus.csv") if (cfg.output_dir / "rex_omitted_temp_focus.csv").exists() else pd.DataFrame()
    loto = pd.read_csv(cfg.output_dir / "loto_model_selection_summary.csv") if (cfg.output_dir / "loto_model_selection_summary.csv").exists() else pd.DataFrame()
    shift = pd.read_csv(cfg.output_dir / "shift_tau_omitted_temp_focus.csv") if (cfg.output_dir / "shift_tau_omitted_temp_focus.csv").exists() else pd.DataFrame()
    lines = [
        "# Pure Unseen-Temperature Extrapolation Robustness Diagnostic",
        "",
        "This report evaluates temperature-domain extrapolation objectives. It does not claim that pure unseen-temperature extrapolation is solved.",
        "",
        "## REx Omitted-Temperature Focus",
    ]
    if len(focus):
        lines.append(focus.sort_values(["experiment", "omitted_MAE_pct"]).head(60).to_markdown(index=False))
    else:
        lines.append("REx results are not available.")
    lines.extend(["", "## Leave-One-Temperature-Out Selection"])
    if len(loto):
        lines.append(loto.head(30).to_markdown(index=False))
    else:
        lines.append("LOTO selection results are not available.")
    lines.extend(["", "## Shift-Tau Corrector"])
    if len(shift):
        lines.append(shift.sort_values(["experiment", "omitted_MAE_pct"]).head(30).to_markdown(index=False))
    else:
        lines.append("Shift-tau corrector results are not available or were not run.")
    lines.extend([
        "",
        "## Safe Interpretation",
        "- REx/shift-tau can be discussed only as attempts to improve omitted-temperature robustness.",
        "- If omitted-temperature MAE improves but seen-temperature MAE worsens, report that tradeoff explicitly.",
        "- Temperature coverage remains critical unless all omitted-temperature folds improve consistently.",
        "",
        "## Forbidden Interpretation",
        "- The model solves pure unseen-temperature extrapolation.",
        "- Test SOC labels were used for OOD/adaptation training.",
        "- Learned voltage components are true physical polarization or true hysteresis.",
    ])
    path = cfg.output_dir / "extrapolation_robustness_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_extrapolation_robustness_suite(
    cfg: CFG | None = None,
    *,
    run_rex=True,
    run_loto=True,
    run_shift_tau=False,
    rex_lambdas=REX_LAMBDAS,
):
    cfg = cfg or make_cfg()
    out = {}
    if run_loto:
        out["loto"] = run_loto_validation(cfg, lambda_rex_values=rex_lambdas)
    elif run_rex:
        out["rex"] = run_rex_experiment(cfg, lambda_rex_values=rex_lambdas)
    if run_shift_tau:
        out["shift_tau"] = run_shift_tau_soc_experiment(cfg)
    out["ood_comparison"] = write_rex_shift_tau_ood_comparison(cfg)
    out["report"] = write_extrapolation_robustness_report(cfg)
    return out
