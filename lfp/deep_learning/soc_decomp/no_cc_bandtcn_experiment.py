from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import make_cfg
from .deep_no_leak_experiment import BRANCH_BAND_DEEP_FEATURES, add_derived_features, CausalConvBlock
from .no_cc_experiment import (
    load_reference_fold_metrics,
    load_reference_temperature_metrics,
    metrics_for_group,
    summarize_focus,
)
from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .training import make_scaled_frames_for_ablation


FOLDS = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50")
FORBIDDEN_INPUT_TOKENS = (
    "SOC",
    "soc",
    "cumulative",
    "Q_ref",
    "q_cutoff",
    "trajectory_fraction",
    "time_index",
    "end_index",
    "trajectory_id",
    "file_name",
)


BANDTCN_VARIANTS = {
    "BandTCN_w150": {"scales": (150,), "objective": "base", "high_temp_weight": 1.0},
    "BandTCN_w300": {"scales": (300,), "objective": "base", "high_temp_weight": 1.0},
    "BandTCN_multiscale_50_150_300": {"scales": (50, 150, 300), "objective": "base", "high_temp_weight": 1.0},
    "BandTCN_multiscale_50_150_500": {"scales": (50, 150, 500), "objective": "base", "high_temp_weight": 1.0},
    "BandTCN_REX": {"scales": (150,), "objective": "rex", "high_temp_weight": 1.0},
    "BandTCN_GroupDRO": {"scales": (150,), "objective": "groupdro", "high_temp_weight": 1.0},
    "BandTCN_REX_GroupDRO": {"scales": (150,), "objective": "rex_groupdro", "high_temp_weight": 1.0},
    "BandTCN_highT": {"scales": (150,), "objective": "base", "high_temp_weight": 2.5},
}


@dataclass
class NoCCBandTCNConfig:
    base_dir: Path = Path(".")
    output_prefix: str = "no_cc_bandtcn"
    folds: tuple[str, ...] = FOLDS
    variants: tuple[str, ...] = (
        "BandTCN_w150",
        "BandTCN_w300",
        "BandTCN_multiscale_50_150_300",
        "BandTCN_multiscale_50_150_500",
        "BandTCN_REX",
        "BandTCN_GroupDRO",
        "BandTCN_REX_GroupDRO",
        "BandTCN_highT",
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 80
    batch_size: int = 2048
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 96
    layers: int = 5
    kernel_size: int = 5
    dropout: float = 0.06
    norm_kind: str = "channel"
    huber_beta: float = 0.02
    lambda_rex: float = 0.30
    lambda_worst: float = 0.30
    lambda_smooth: float = 0.00
    stride: int = 3
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 20
    force: bool = False
    save_predictions: bool = True


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def audit_feature_columns(cols: list[str]):
    bad = []
    for col in cols:
        for tok in FORBIDDEN_INPUT_TOKENS:
            if tok.lower() in col.lower():
                bad.append(col)
                break
    if bad:
        raise RuntimeError(f"CUMULATIVE_FEATURE_LEAK: forbidden BandTCN input columns selected: {sorted(set(bad))}")


def band_feature_columns() -> list[str]:
    cols = list(BRANCH_BAND_DEEP_FEATURES)
    audit_feature_columns(cols)
    return cols


def run_key(prefix: str, variant: str, fold: str, seed: int) -> str:
    safe_fold = fold.replace(" ", "_").replace("-", "N")
    return f"{prefix}_{variant}_{safe_fold}_seed{seed}"


def run_paths(cfg: NoCCBandTCNConfig, variant: str, fold: str, seed: int) -> dict[str, Path]:
    run_dir = cfg.base_dir / "no_cc_bandtcn_runs" / run_key(cfg.output_prefix, variant, fold, seed)
    return {
        "dir": run_dir,
        "pred": run_dir / "prediction_rows.csv.gz",
        "history": run_dir / "history.csv",
        "status": run_dir / "status.json",
    }


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def status_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "done"
    except Exception:
        return False


def frame_observability(frame: pd.DataFrame, start: int, end: int) -> dict[str, float]:
    sl = slice(start, end + 1)
    i = frame["I_raw"].to_numpy(np.float64)[sl]
    di = frame["dI"].to_numpy(np.float64)[sl]
    vc = frame["V_corr_raw"].to_numpy(np.float64)[sl]
    vres = (frame["V_raw"].to_numpy(np.float64) - frame["V_corr_raw"].to_numpy(np.float64))[sl]
    band_cols = [c for c in ["V_residual_low", "V_residual_mid", "V_residual_high"] if c in frame.columns]
    if band_cols:
        band = frame[band_cols].to_numpy(np.float64)[sl]
        branch_band_energy = float(np.mean(np.sum(band ** 2, axis=1)))
    else:
        branch_band_energy = float(np.mean(vres ** 2))
    abs_i = np.abs(i)
    rest = abs_i <= 0.05
    transitions = int(np.sum(np.abs(di) > 0.05))
    relaxation = float(np.mean(np.abs(np.diff(vc[rest])))) if int(rest.sum()) > 2 else 0.0
    return {
        "current_transition_count": transitions,
        "dI_energy": float(np.mean(di ** 2)),
        "voltage_response_amplitude": float(np.nanpercentile(vc, 95) - np.nanpercentile(vc, 5)),
        "relaxation_visibility": relaxation,
        "branch_band_energy": branch_band_energy,
        "V_corr_slope": float(np.mean(np.abs(np.diff(vc)))) if len(vc) > 1 else 0.0,
        "endpoint_absI": float(abs_i[-1]) if len(abs_i) else np.nan,
    }


class BandWindowDataset(Dataset):
    def __init__(self, scaled_frames, raw_frames, feature_cols, window_len: int, stride: int):
        self.feature_cols = list(feature_cols)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.frames = []
        self.raw_frames = []
        self.index = []
        for fi, (scaled, raw) in enumerate(zip(scaled_frames, raw_frames)):
            s = scaled.reset_index(drop=True)
            r = raw.reset_index(drop=True)
            cache = {
                "x": np.ascontiguousarray(s[self.feature_cols].to_numpy(np.float32)),
                "y": np.ascontiguousarray(s["SOC_physical"].to_numpy(np.float32)),
                "file_name": s["file_name"].to_numpy(),
                "trajectory_id": s["trajectory_id"].to_numpy(),
                "end_index": s["end_index"].to_numpy(np.int64),
                "temperature": s["temperature"].to_numpy(np.float32),
                "drive_cycle": s["drive_cycle"].to_numpy(),
            }
            self.frames.append(cache)
            self.raw_frames.append(r)
            n = len(s)
            if n < self.window_len:
                continue
            for start in range(0, n - self.window_len + 1, self.stride):
                end = start + self.window_len - 1
                if np.isfinite(cache["y"][end]):
                    self.index.append((fi, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        x = f["x"][start:end + 1]
        y = np.array([f["y"][end]], dtype=np.float32)
        obs = frame_observability(self.raw_frames[fi], start, end)
        soc_val = float(f["y"][end])
        if soc_val < 0.2:
            soc_bin = 0
        elif soc_val <= 0.8:
            soc_bin = 1
        else:
            soc_bin = 2
        meta = {
            "file_name": f["file_name"][end],
            "trajectory_id": f["trajectory_id"][end],
            "end_index": int(f["end_index"][end]),
            "temperature": float(f["temperature"][end]),
            "drive_cycle": f["drive_cycle"][end],
            "soc_bin": int(soc_bin),
            **obs,
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


def endpoint_temperatures(ds: BandWindowDataset) -> np.ndarray:
    return np.asarray([ds.frames[fi]["temperature"][end] for fi, _, end in ds.index], dtype=np.float32)


def make_loader(ds: BandWindowDataset, cfg: NoCCBandTCNConfig, *, shuffle: bool, high_temp_weight: float = 1.0):
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    if not shuffle:
        return DataLoader(ds, shuffle=False, **kwargs)
    temps = endpoint_temperatures(ds)
    unique, counts = np.unique(temps, return_counts=True)
    count_map = {float(t): int(c) for t, c in zip(unique, counts)}
    weights = np.asarray([1.0 / count_map[float(t)] for t in temps], dtype=np.float64)
    if float(high_temp_weight) != 1.0:
        weights = weights * np.where(temps >= 40.0, float(high_temp_weight), 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(ds, sampler=sampler, **kwargs)


class TCNBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, layers: int, kernel_size: int, dropout: float, norm_kind: str):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(*[
            CausalConvBlock(hidden_size, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout, norm_kind=norm_kind)
            for i in range(int(layers))
        ])

    def forward(self, x):
        h = self.input_proj(x.transpose(1, 2))
        h = self.blocks(h)
        return h[:, :, -1]


def augment_window(x: torch.Tensor) -> torch.Tensor:
    delta = x - x[:, :1, :]
    # Window-local normalized position is allowed. Absolute start/end timestep is not appended.
    t = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1).expand(x.shape[0], -1, 1)
    return torch.cat([x, delta, t], dim=-1)


class MultiScaleBandTCN(nn.Module):
    def __init__(self, input_dim: int, scales: tuple[int, ...], cfg: NoCCBandTCNConfig):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        aug_dim = int(input_dim) * 2 + 1
        self.branches = nn.ModuleList([
            TCNBranch(aug_dim, cfg.hidden_size, cfg.layers, cfg.kernel_size, cfg.dropout, cfg.norm_kind)
            for _ in self.scales
        ])
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size * len(self.scales), cfg.hidden_size),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(cfg.hidden_size, 1),
        )

    def forward(self, x):
        hs = []
        for scale, branch in zip(self.scales, self.branches):
            xs = x[:, -scale:, :]
            hs.append(branch(augment_window(xs)))
        h = torch.cat(hs, dim=1)
        return torch.sigmoid(self.head(h))


def move_float(x):
    return x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def meta_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().tolist()
    return list(values)


def objective_loss(pred, y, meta, cfg: NoCCBandTCNConfig, objective: str):
    sample = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
    temps = [float(v) for v in meta_list(meta["temperature"])]
    groups = []
    for temp in sorted(set(temps)):
        idx = [i for i, t in enumerate(temps) if t == temp]
        groups.append(sample.index_select(0, torch.as_tensor(idx, device=pred.device, dtype=torch.long)).mean())
    stack = torch.stack(groups) if groups else sample.mean().view(1)
    mean_loss = stack.mean()
    rex = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
    worst = stack.max() if len(stack) > 1 else stack.new_tensor(0.0)
    total = mean_loss
    if objective in {"rex", "rex_groupdro"}:
        total = total + float(cfg.lambda_rex) * rex
    if objective in {"groupdro", "rex_groupdro"}:
        total = total + float(cfg.lambda_worst) * worst
    return total, mean_loss, rex, worst


def collate_meta(meta: dict) -> pd.DataFrame:
    out = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy().tolist()
        else:
            out[k] = list(v)
    return pd.DataFrame(out)


def train_one(variant: str, fold: str, seed: int, cfg: NoCCBandTCNConfig, lookup: pd.DataFrame):
    paths = run_paths(cfg, variant, fold, seed)
    if status_done(paths["status"]) and paths["pred"].exists() and not cfg.force:
        return {"status": "cached", "variant": variant, "fold_name": fold, "seed": seed}
    paths["dir"].mkdir(parents=True, exist_ok=True)
    started = time.time()
    set_seed(seed)
    spec = BANDTCN_VARIANTS[variant]
    scales = tuple(spec["scales"])
    window_len = max(scales)
    base_cfg = make_cfg()
    base_cfg.output_dir = cfg.base_dir
    base_cfg.base_dir = cfg.base_dir
    base_cfg = experiment_cfg(base_cfg, fold)
    full_branch_dir = cfg.base_dir / "decomposed_features_train_temp_minus10_0_10_20_25_50"
    if full_branch_dir.exists():
        base_cfg.decomposed_dir = full_branch_dir
    configure_strict_training(base_cfg)
    base_cfg.window_len = window_len
    base_cfg.stride = int(cfg.stride)
    base_cfg.batch_size = int(cfg.batch_size)
    raw_frames = add_derived_features(load_relabelled_frames(base_cfg, fold, lookup))
    cols = band_feature_columns()
    available = set().union(*(set(f.columns) for split in raw_frames.values() for f in split))
    missing = [c for c in cols if c not in available]
    if missing:
        raise KeyError(f"{variant} {fold}: missing columns {missing}")
    scaled, _ = make_scaled_frames_for_ablation(raw_frames, cols)
    train_ds = BandWindowDataset(scaled["train"], raw_frames["train"], cols, window_len, cfg.stride)
    test_ds = BandWindowDataset(scaled["test"], raw_frames["test"], cols, window_len, cfg.stride)
    if len(train_ds) == 0:
        raise ValueError(f"{variant} {fold}: empty train dataset")
    train_loader = make_loader(train_ds, cfg, shuffle=True, high_temp_weight=float(spec.get("high_temp_weight", 1.0)))
    test_loader = make_loader(test_ds, cfg, shuffle=False)
    model = MultiScaleBandTCN(len(cols), scales, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    hist = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        means = []
        rexes = []
        worsts = []
        for x, y, meta in train_loader:
            x = move_float(x)
            y = move_float(y)
            pred = model(x)
            loss, mean_loss, rex, worst = objective_loss(pred, y, meta, cfg, str(spec["objective"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            means.append(float(mean_loss.detach().cpu()))
            rexes.append(float(rex.detach().cpu()))
            worsts.append(float(worst.detach().cpu()))
        row = {
            "variant": variant,
            "fold_name": fold,
            "seed": seed,
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "mean_huber": float(np.mean(means)),
            "rex_var": float(np.mean(rexes)),
            "worst_temp_loss": float(np.mean(worsts)),
            "objective": spec["objective"],
            "scales": "+".join(str(s) for s in scales),
        }
        hist.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(f"{variant} {fold} seed={seed} ep={ep} loss={row['train_loss']:.5f}", flush=True)
    pred_rows = []
    model.eval()
    with torch.no_grad():
        for x, y, meta in test_loader:
            p = model(move_float(x)).detach().cpu().numpy()[:, 0]
            yy = y.numpy()[:, 0]
            mdf = collate_meta(meta)
            mdf["model_name"] = variant
            mdf["run_model_name"] = variant
            mdf["target_label"] = "physical"
            mdf["label_type"] = "physical"
            mdf["fold_name"] = fold
            mdf["experiment"] = fold
            mdf["seed"] = int(seed)
            mdf["y_true"] = yy
            mdf["y_pred"] = p
            pred_rows.append(mdf)
    pred = pd.concat(pred_rows, ignore_index=True)
    pred["temperature_C"] = pred["temperature"].astype(float)
    pred["time_index"] = pred["end_index"]
    pred["error"] = pred["y_pred"] - pred["y_true"]
    pred["abs_error"] = np.abs(pred["error"])
    max_index = pred.groupby("trajectory_id")["end_index"].transform("max").replace(0, np.nan)
    pred["trajectory_fraction"] = pred["end_index"] / max_index
    pred["is_plateau_20_80"] = (pred["y_true"] >= 0.2) & (pred["y_true"] <= 0.8)
    pred["is_cutoff_last10"] = pred["trajectory_fraction"] >= 0.9
    pred["label_policy"] = "physical_smoothQ"
    keep = [
        "file_name", "trajectory_id", "end_index", "temperature", "temperature_C", "drive_cycle",
        "current_transition_count", "dI_energy", "voltage_response_amplitude", "relaxation_visibility",
        "branch_band_energy", "V_corr_slope", "endpoint_absI",
        "model_name", "run_model_name", "target_label", "label_type", "fold_name", "experiment", "seed",
        "y_true", "y_pred", "time_index", "error", "abs_error", "trajectory_fraction",
        "is_plateau_20_80", "is_cutoff_last10", "label_policy", "soc_bin",
    ]
    pred = pred[[c for c in keep if c in pred.columns]]
    pred.to_csv(paths["pred"], index=False, compression="gzip")
    pd.DataFrame(hist).to_csv(paths["history"], index=False)
    write_json(paths["status"], {
        "status": "done",
        "variant": variant,
        "fold_name": fold,
        "seed": seed,
        "duration_s": time.time() - started,
        "train_windows": len(train_ds),
        "test_windows": len(test_ds),
        "feature_cols": cols,
        "scales": scales,
        "objective": spec["objective"],
        "strict_no_cc": True,
        "absolute_timestep_input": False,
        "window_local_position_input": True,
        "window_position_policy": "uses normalized within-window position t in [0,1]; no absolute start/end timestep or trajectory progress is appended to model input",
    })
    return {"status": "done", "variant": variant, "fold_name": fold, "seed": seed}


def fold_target_type(fold: str) -> str:
    if fold == "Exp D":
        return "included_diagnostic"
    if fold in {"Omit N10", "Omit 50"}:
        return "outside"
    return "omitted"


def compute_tables(pred: pd.DataFrame):
    result_rows = []
    temp_rows = []
    obs_rows = []
    for (model, seed, fold), g in pred.groupby(["model_name", "seed", "fold_name"]):
        target_temp = float(EXPERIMENTS[fold]["omitted_temp_C"])
        tg = g[np.isclose(g["temperature_C"].astype(float), target_temp)]
        if len(tg):
            row = {
                "model_name": model,
                "seed": int(seed),
                "fold_name": fold,
                "target_temperature_C": target_temp,
                "target_type": fold_target_type(fold),
                "source": "no_cc_bandtcn",
            }
            row.update(metrics_for_group(tg))
            result_rows.append(row)
        for temp, gg in g.groupby("temperature_C"):
            row = {
                "model_name": model,
                "seed": int(seed),
                "fold_name": fold,
                "temperature_C": float(temp),
                "source": "no_cc_bandtcn",
            }
            row.update(metrics_for_group(gg))
            temp_rows.append(row)
        gg = g.copy()
        if "branch_band_energy" in gg.columns:
            try:
                gg["observability_bin"] = pd.qcut(gg["branch_band_energy"].rank(method="first"), q=3, labels=["low", "mid", "high"])
            except ValueError:
                gg["observability_bin"] = "all"
            for obs_bin, og in gg.groupby("observability_bin", observed=False):
                if len(og):
                    row = {
                        "model_name": model,
                        "seed": int(seed),
                        "fold_name": fold,
                        "observability_bin": str(obs_bin),
                        "mean_branch_band_energy": float(og["branch_band_energy"].mean()),
                    }
                    row.update(metrics_for_group(og))
                    obs_rows.append(row)
    results = pd.DataFrame(result_rows)
    by_temp = pd.DataFrame(temp_rows)
    obs = pd.DataFrame(obs_rows)
    focus = summarize_focus(results) if len(results) else pd.DataFrame()
    return results, by_temp, focus, obs


OBSERVABILITY_FEATURES = (
    "current_transition_count",
    "dI_energy",
    "voltage_response_amplitude",
    "relaxation_visibility",
    "branch_band_energy",
    "V_corr_slope",
)


def make_observability_vs_error(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for (model, seed, fold), g in pred.groupby(["model_name", "seed", "fold_name"]):
        for feature in OBSERVABILITY_FEATURES:
            if feature not in g.columns:
                continue
            gg = g.dropna(subset=[feature, "abs_error"]).copy()
            if gg.empty:
                continue
            try:
                gg["observability_bin"] = pd.qcut(
                    gg[feature].rank(method="first"),
                    q=3,
                    labels=["low", "mid", "high"],
                )
            except ValueError:
                gg["observability_bin"] = "all"
            corr = gg[[feature, "abs_error"]].corr(method="spearman").iloc[0, 1]
            for obs_bin, og in gg.groupby("observability_bin", observed=False):
                if og.empty:
                    continue
                row = {
                    "model_name": model,
                    "seed": int(seed),
                    "fold_name": fold,
                    "observability_feature": feature,
                    "observability_bin": str(obs_bin),
                    "feature_mean": float(og[feature].mean()),
                    "feature_median": float(og[feature].median()),
                    "abs_error_spearman": float(corr) if pd.notna(corr) else np.nan,
                }
                row.update(metrics_for_group(og))
                rows.append(row)
    return pd.DataFrame(rows)


def write_observability_extra_outputs(
    cfg: NoCCBandTCNConfig,
    pred: pd.DataFrame,
    tg_pred: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    base = cfg.base_dir
    combined = pd.concat([pred, tg_pred], ignore_index=True, sort=False) if len(tg_pred) else pred.copy()
    obs_vs_error = make_observability_vs_error(combined)
    obs_vs_error.to_csv(base / "no_cc_observability_vs_error.csv", index=False)

    lines = [
        "# Strict NoCC Observability And Uncertainty Report",
        "",
        "## Scope",
        "- This is a post-hoc diagnostic computed from saved predictions.",
        "- Observability features are derived from V/I/T window response, not from SOC, cumulative Ah, or current-integrated SOC state.",
        "- Current is used only as instantaneous excitation, not integrated into SOC state.",
        "",
    ]
    if len(obs_vs_error):
        metric_cols = [
            c for c in [
                "MAE_pct",
                "RMSE_pct",
                "catastrophic_error_rate_5pct",
                "n_samples",
            ]
            if c in obs_vs_error.columns
        ]
        summary = (
            obs_vs_error.groupby(["model_name", "observability_feature", "observability_bin"], observed=False)
            [metric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        lines += ["## Error By Observability Bin", summary.to_markdown(index=False), ""]
        corr = (
            obs_vs_error.groupby(["model_name", "observability_feature"])["abs_error_spearman"]
            .mean()
            .reset_index()
            .sort_values(["model_name", "abs_error_spearman"], ascending=[True, False])
        )
        lines += ["## Spearman Correlation With Absolute Error", corr.to_markdown(index=False), ""]
    else:
        lines += ["## Error By Observability Bin", "No prediction rows were available.", ""]

    if len(tg_pred) and len(weights):
        keys = ["fold_name", "seed", "trajectory_id", "end_index"]
        risk_cols = ["expert_disagreement", "ood_proxy", "recent_prediction_jitter"]
        merged = tg_pred[keys + ["abs_error"]].merge(weights[keys + risk_cols], on=keys, how="inner")
        risk_rows = []
        for col in risk_cols:
            if col not in merged:
                continue
            gg = merged.dropna(subset=[col, "abs_error"]).copy()
            if gg.empty:
                continue
            try:
                gg["risk_bin"] = pd.qcut(gg[col].rank(method="first"), q=4, labels=["q1_low", "q2", "q3", "q4_high"])
            except ValueError:
                gg["risk_bin"] = "all"
            corr = gg[[col, "abs_error"]].corr(method="spearman").iloc[0, 1]
            for risk_bin, og in gg.groupby("risk_bin", observed=False):
                risk_rows.append({
                    "uncertainty_proxy": col,
                    "risk_bin": str(risk_bin),
                    "proxy_mean": float(og[col].mean()),
                    "MAE_pct": float(og["abs_error"].mean()),
                    "abs_error_spearman": float(corr) if pd.notna(corr) else np.nan,
                    "n_samples": int(len(og)),
                })
        risk_df = pd.DataFrame(risk_rows)
        if len(risk_df):
            lines += ["## ThermoGuard Label-Free Risk Proxies", risk_df.to_markdown(index=False), ""]

    lines += [
        "## Interpretation",
        "- Low-observability bins with higher MAE indicate regions where voltage/dynamic response is insufficient for strict NoCC inference.",
        "- Positive uncertainty-proxy correlation means the label-free guard signal tracks larger error regions.",
        "- These diagnostics do not prove NoCC solves extrapolation; they only identify where the strict ablation is more or less observable.",
    ]
    (base / "no_cc_uncertainty_report.md").write_text("\n".join(lines), encoding="utf-8")


def recent_jitter(df: pd.DataFrame, pred_col: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.sort_values("end_index").groupby("trajectory_id").groups.items():
        g = df.loc[idx].sort_values("end_index")
        d = g[pred_col].astype(float).diff().abs()
        out.loc[g.index] = d.rolling(25, min_periods=3).mean().bfill().fillna(d.mean()).to_numpy()
    return out.fillna(0.0)


def build_thermoguard(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["BandTCN_w150", "BandTCN_REX", "BandTCN_highT"]
    if pred.empty or not set(required).issubset(set(pred["model_name"].unique())):
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    weights_rows = []
    keys = ["fold_name", "seed", "trajectory_id", "end_index"]
    meta_cols = [
        "file_name", "temperature", "temperature_C", "drive_cycle", "target_label", "label_type",
        "experiment", "y_true", "time_index", "trajectory_fraction", "is_plateau_20_80",
        "is_cutoff_last10", "soc_bin", "endpoint_absI", "dI_energy",
        "voltage_response_amplitude", "branch_band_energy", "V_corr_slope",
    ]
    for (fold, seed), fg in pred.groupby(["fold_name", "seed"]):
        base = fg[fg["model_name"].eq("BandTCN_w150")][keys + meta_cols + ["y_pred"]].rename(columns={"y_pred": "p_general"})
        rex = fg[fg["model_name"].eq("BandTCN_REX")][keys + ["y_pred"]].rename(columns={"y_pred": "p_rex"})
        high = fg[fg["model_name"].eq("BandTCN_highT")][keys + ["y_pred"]].rename(columns={"y_pred": "p_highT"})
        m = base.merge(rex, on=keys, how="inner").merge(high, on=keys, how="inner")
        if m.empty:
            continue
        preds = m[["p_general", "p_rex", "p_highT"]].to_numpy(float)
        disagreement = np.mean(np.abs(preds - preds.mean(axis=1, keepdims=True)), axis=1)
        temp = m["temperature_C"].astype(float).to_numpy()
        high_gate = 1.0 / (1.0 + np.exp(-(temp - 40.0) / 3.0))
        ood = (m["branch_band_energy"].astype(float) - m["branch_band_energy"].astype(float).median()).abs()
        ood = ood / (ood.quantile(0.9) + 1e-12)
        jitter_general = recent_jitter(m.assign(y_pred=m["p_general"]), "y_pred").to_numpy(float)
        risk = np.clip(0.5 * disagreement / (np.nanpercentile(disagreement, 90) + 1e-12) + 0.3 * ood + 0.2 * jitter_general / (np.nanpercentile(jitter_general, 90) + 1e-12), 0, 2)
        w_high = np.clip(0.15 + 0.65 * high_gate, 0.05, 0.80)
        w_rex = np.clip(0.20 + 0.40 * risk, 0.10, 0.70)
        w_general = np.clip(1.0 - w_high - w_rex, 0.05, 0.85)
        total = w_general + w_rex + w_high
        w_general, w_rex, w_high = w_general / total, w_rex / total, w_high / total
        y_pred = w_general * m["p_general"].to_numpy(float) + w_rex * m["p_rex"].to_numpy(float) + w_high * m["p_highT"].to_numpy(float)
        out = m.copy()
        out["model_name"] = "NoCC_ThermoGuard_BandTCN"
        out["run_model_name"] = out["model_name"]
        out["y_pred"] = y_pred
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        out["label_policy"] = "physical_smoothQ"
        rows.append(out)
        wr = out[keys + ["temperature_C"]].copy()
        wr["w_general"] = w_general
        wr["w_rex"] = w_rex
        wr["w_highT"] = w_high
        wr["expert_disagreement"] = disagreement
        wr["ood_proxy"] = ood
        wr["recent_prediction_jitter"] = jitter_general
        weights_rows.append(wr)
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.concat(weights_rows, ignore_index=True) if weights_rows else pd.DataFrame(),
    )


def load_completed_predictions(cfg: NoCCBandTCNConfig) -> pd.DataFrame:
    chunks = []
    for variant in cfg.variants:
        for fold in cfg.folds:
            for seed in cfg.seeds:
                paths = run_paths(cfg, variant, fold, int(seed))
                if status_done(paths["status"]) and paths["pred"].exists():
                    chunks.append(pd.read_csv(paths["pred"]))
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()


def write_reports(cfg: NoCCBandTCNConfig, results, by_temp, focus, obs, tg_results, tg_focus, weights):
    base = cfg.base_dir
    lines = [
        "# Strict NoCC BandTCN Report",
        "",
        "## Policy",
        "- No SOC input, no usable SOC input, no SOC_CC input.",
        "- No cumulative Ah, integrated-current feature, trajectory progress, absolute time index, trajectory ID, or shifted target input.",
        "- Window-local normalized position is allowed; absolute start/end timestep is not used as input.",
        "- Current is used only as instantaneous excitation through `I_raw`, `dI`, `absI`, and voltage decomposition/band response features.",
        "- Model is a stateless fixed-window causal TCN endpoint SOC estimator.",
        "",
    ]
    if len(focus):
        lines += ["## BandTCN Focus", focus.sort_values(["scope", "average_MAE_pct"]).to_markdown(index=False), ""]
    if len(tg_focus):
        lines += ["## NoCC-ThermoGuard Focus", tg_focus.sort_values(["scope", "average_MAE_pct"]).to_markdown(index=False), ""]
    if len(obs):
        obs_sum = obs.groupby(["model_name", "observability_bin"])[["MAE_pct", "n_samples"]].mean(numeric_only=True).reset_index()
        lines += ["## Observability", obs_sum.to_markdown(index=False), ""]
    lines += [
        "## Safe Conclusion",
        "This result is a strict NoCC ablation/rebuild. It should not be used to claim that current integration is unnecessary or that temperature extrapolation is solved.",
    ]
    (base / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")
    if len(tg_focus):
        (base / "no_cc_thermoguard_report.md").write_text("\n".join([
            "# NoCC-ThermoGuard BandTCN Report",
            "",
            tg_focus.sort_values(["scope", "average_MAE_pct"]).to_markdown(index=False),
            "",
            "The guard is label-free and uses temperature, expert disagreement, branch-band energy proxy, and recent prediction jitter.",
        ]), encoding="utf-8")


def final_comparison(cfg: NoCCBandTCNConfig, band_results: pd.DataFrame, tg_results: pd.DataFrame):
    rows = []
    refs = load_reference_fold_metrics(cfg.base_dir)
    if len(refs):
        refs = refs[refs["model_name"].isin(["A3_V_raw_I_T", "R5_GATED_AUG_REX"])].copy()
        refs["comparison_status"] = "reference_10seed_mean"
        rows.append(refs)
    no_cc_path = cfg.base_dir / "no_cc_results.csv"
    if no_cc_path.exists():
        no = pd.read_csv(no_cc_path)
        no = no[(no["source"].eq("no_cc_one_seed")) & (no["model_name"].eq("NoCC_DynamicResponse"))].copy()
        no["comparison_status"] = "strict_no_cc_dynamic_response"
        rows.append(no)
    if len(band_results):
        b = band_results.copy()
        b["comparison_status"] = "strict_no_cc_bandtcn"
        rows.append(b)
    if len(tg_results):
        tg = tg_results.copy()
        tg["comparison_status"] = "strict_no_cc_thermoguard"
        rows.append(tg)
    comp = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    comp.to_csv(cfg.base_dir / "no_cc_final_comparison.csv", index=False)
    if len(comp):
        focus_input = comp.copy()
        focus_input["source"] = focus_input["comparison_status"]
        focus = summarize_focus(focus_input)
    else:
        focus = pd.DataFrame()
    lines = [
        "# Strict NoCC Final Comparison",
        "",
        "Old NeuralECM/ThermoGuard current-integration results are invalidated as main comparators and are not included as main rows here.",
        "",
    ]
    if len(focus):
        lines += [focus.sort_values(["scope", "average_MAE_pct"]).to_markdown(index=False), ""]
    lines += [
        "## Required Answers",
        "1. Strict NoCC should be compared against A3/R5 leak-free baselines and NoCC BandTCN, not old CC-assisted NeuralECM as main evidence.",
        "2. The largest degradation should be read from outside-range rows (`Omit N10`, `Omit 50`).",
        "3. Branch-bands close part of the gap only if their target-fold rows approach the previous best without SOC/cumulative inputs.",
        "4. Robust objectives are judged by worst target-fold MAE and outside average/worst MAE.",
        "5. NoCC-ThermoGuard is viable only if it improves outside-range without tuning on test labels.",
        "6. Safe wording: current is used only as instantaneous excitation, not integrated into SOC state.",
    ]
    (cfg.base_dir / "no_cc_final_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    return comp


def aggregate_and_write(cfg: NoCCBandTCNConfig):
    pred = load_completed_predictions(cfg)
    results, by_temp, focus, obs = compute_tables(pred) if len(pred) else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.base_dir / "no_cc_bandtcn_results.csv", index=False)
    by_temp.to_csv(cfg.base_dir / "no_cc_bandtcn_by_temperature.csv", index=False)
    focus.to_csv(cfg.base_dir / "no_cc_bandtcn_focus.csv", index=False)
    obs.to_csv(cfg.base_dir / "no_cc_observability_metrics.csv", index=False)
    if cfg.save_predictions and len(pred):
        pred.to_csv(cfg.base_dir / "no_cc_bandtcn_prediction_rows.csv.gz", index=False, compression="gzip")
    domain = results[results["model_name"].isin(["BandTCN_REX", "BandTCN_GroupDRO", "BandTCN_REX_GroupDRO"])] if len(results) else pd.DataFrame()
    domain.to_csv(cfg.base_dir / "no_cc_bandtcn_domain_robust_summary.csv", index=False)
    if len(results):
        results[results["model_name"].eq("BandTCN_REX")].to_csv(cfg.base_dir / "no_cc_bandtcn_rex_results.csv", index=False)
        results[results["model_name"].eq("BandTCN_GroupDRO")].to_csv(cfg.base_dir / "no_cc_bandtcn_groupdro_results.csv", index=False)
    tg_pred, weights = build_thermoguard(pred)
    tg_results, tg_temp, tg_focus, _ = compute_tables(tg_pred) if len(tg_pred) else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    tg_results.to_csv(cfg.base_dir / "no_cc_thermoguard_results.csv", index=False)
    tg_temp.to_csv(cfg.base_dir / "no_cc_thermoguard_by_temperature.csv", index=False)
    tg_focus.to_csv(cfg.base_dir / "no_cc_thermoguard_focus.csv", index=False)
    weights.to_csv(cfg.base_dir / "no_cc_thermoguard_expert_weights.csv", index=False)
    write_observability_extra_outputs(cfg, pred, tg_pred, weights)
    write_reports(cfg, results, by_temp, focus, obs, tg_results, tg_focus, weights)
    final_comparison(cfg, results, tg_results)
    metadata = {
        **asdict(cfg),
        "feature_cols": band_feature_columns(),
        "strict_no_cc": True,
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
        "amp_used": False,
        "device": str(device),
    }
    write_json(cfg.base_dir / "no_cc_bandtcn_metadata.json", metadata)
    try:
        from .strict_no_cc_artifact_manifest import generate_manifest

        generate_manifest(cfg.base_dir)
    except Exception as exc:
        write_json(cfg.base_dir / "strict_no_cc_artifact_manifest_error.json", {
            "status": "failed",
            "error": repr(exc),
        })
    return pred, results, by_temp, focus


def run_all(cfg: NoCCBandTCNConfig):
    cfg.base_dir = Path(cfg.base_dir)
    configure_torch_runtime()
    lookup = load_smoothq_lookup(cfg.base_dir)
    statuses = []
    for seed in cfg.seeds:
        for fold in cfg.folds:
            for variant in cfg.variants:
                st = train_one(str(variant), str(fold), int(seed), cfg, lookup)
                statuses.append(st)
                pd.DataFrame(statuses).to_csv(cfg.base_dir / "no_cc_bandtcn_run_status_log.csv", index=False)
                aggregate_and_write(cfg)
    return aggregate_and_write(cfg)


def parse_args():
    p = argparse.ArgumentParser(description="Strict NoCC branch-band BandTCN experiments.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--output-prefix", default="no_cc_bandtcn")
    p.add_argument("--folds", nargs="+", default=list(FOLDS), choices=list(FOLDS))
    p.add_argument("--variants", nargs="+", default=list(NoCCBandTCNConfig.variants))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = NoCCBandTCNConfig(
        base_dir=args.base_dir,
        output_prefix=args.output_prefix,
        folds=tuple(args.folds),
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        hidden_size=args.hidden_size,
        layers=args.layers,
        num_workers=args.num_workers,
        print_every=args.print_every,
        force=bool(args.force),
    )
    if args.aggregate_only:
        aggregate_and_write(cfg)
    else:
        run_all(cfg)


if __name__ == "__main__":
    main()
