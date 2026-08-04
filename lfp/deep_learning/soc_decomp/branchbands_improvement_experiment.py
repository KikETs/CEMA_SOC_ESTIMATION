from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import inspect
import json
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import make_cfg
from .deep_no_leak_experiment import (
    BRANCH_BAND_DEEP_FEATURES,
    BRANCH_DEEP_FEATURES,
    DeepNoLeakTCN,
    add_derived_features,
    temp_balanced_rex_loss,
)
from .models import collate_meta_to_frame
from .no_cc_bandtcn_experiment import frame_observability
from .no_cc_experiment import metrics_for_group, summarize_focus
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
FORBIDDEN_INPUT_PATTERNS = (
    re.compile(r"SOC_CC", re.I),
    re.compile(r"SOC_usable", re.I),
    re.compile(r"SOC_physical", re.I),
    re.compile(r"cumulative", re.I),
    re.compile(r"Q_ref", re.I),
    re.compile(r"q_cutoff", re.I),
    re.compile(r"\bAh\b", re.I),
    re.compile(r"trajectory_fraction", re.I),
    re.compile(r"time_index", re.I),
    re.compile(r"end_index", re.I),
    re.compile(r"trajectory_id", re.I),
    re.compile(r"file_name", re.I),
)

COLD_EMA_FEATURES = [
    "V_corr_raw_ema200",
    "V_corr_raw_ema800",
    "V_raw_dev_ema200",
    "V_raw_dev_ema800",
    "V_pol_raw_ema200",
    "V_pol_raw_ema800",
    "V_pol_raw_dev_ema200",
    "V_pol_raw_dev_ema800",
    "R0_ema200",
    "R0_ema800",
]

LOCAL_RESIDUAL_FEATURES = [
    ("V_residual_raw_window", "V_residual_raw"),
    ("V_residual_low_window", "V_residual_low"),
    ("V_residual_mid_window", "V_residual_mid"),
    ("V_residual_high_window", "V_residual_high"),
    ("V_residual_low_x_T_window", "V_residual_low_x_T"),
    ("V_residual_mid_x_absI_window", "V_residual_mid_x_absI"),
    ("V_residual_high_x_abs_dI_window", "V_residual_high_x_abs_dI"),
]

HYBRID_RESIDUAL_FEATURES = [
    ("V_residual_raw_window", "V_residual_raw"),
    ("V_residual_mid_window", "V_residual_mid"),
    ("V_residual_high_window", "V_residual_high"),
    ("V_residual_mid_x_absI_window", "V_residual_mid_x_absI"),
    ("V_residual_high_x_abs_dI_window", "V_residual_high_x_abs_dI"),
]

VARIANTS = {
    "BandTCN_w150_base": {
        "residual_mode": "trajectory_causal",
        "cold_ema": False,
        "cold_head_limit": None,
    },
    "BandTCN_w150_coldEMA": {
        "residual_mode": "trajectory_causal",
        "cold_ema": True,
        "cold_head_limit": None,
    },
    "BandTCN_w150_coldEMA_coldHead_lim0p02": {
        "residual_mode": "trajectory_causal",
        "cold_ema": True,
        "cold_head_limit": 0.02,
    },
    "BandTCN_w150_coldEMA_coldHead_lim0p03": {
        "residual_mode": "trajectory_causal",
        "cold_ema": True,
        "cold_head_limit": 0.03,
    },
    "BandTCN_w150_coldEMA_coldHead_lim0p05": {
        "residual_mode": "trajectory_causal",
        "cold_ema": True,
        "cold_head_limit": 0.05,
    },
    "BandTCN_w150_residual_windowLocal": {
        "residual_mode": "window_local",
        "cold_ema": False,
        "cold_head_limit": None,
    },
    "BandTCN_w150_residual_hybrid": {
        "residual_mode": "hybrid",
        "cold_ema": False,
        "cold_head_limit": None,
    },
}


@dataclass
class BranchBandsImprovementConfig:
    base_dir: Path = Path(".")
    output_prefix: str = "branchbands_improvement"
    folds: tuple[str, ...] = FOLDS
    variants: tuple[str, ...] = tuple(VARIANTS.keys())
    seeds: tuple[int, ...] = (0,)
    feature_dir: str = "decomposed_features_train_temp_minus10_0_10_20_25_50"
    window_len: int = 150
    stride: int = 3
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 128
    layers: int = 6
    kernel_size: int = 5
    norm_kind: str = "channel"
    dropout: float = 0.04
    lambda_rex: float = 2.0
    rex_group: str = "temperature_drive"
    lambda_smooth: float = 0.0
    endpoint_loss_weight: float = 0.0
    lambda_worst: float = 0.0
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 25
    force: bool = False
    save_predictions: bool = True


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def status_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "done"
    except Exception:
        return False


def safe_fold_name(fold: str) -> str:
    return fold.replace(" ", "_").replace("-", "N")


def run_key(prefix: str, variant: str, fold: str, seed: int) -> str:
    return f"{prefix}_{variant}_{safe_fold_name(fold)}_seed{seed}"


def run_paths(cfg: BranchBandsImprovementConfig, variant: str, fold: str, seed: int) -> dict[str, Path]:
    run_dir = cfg.base_dir / "branchbands_improvement_runs" / run_key(cfg.output_prefix, variant, fold, seed)
    return {
        "dir": run_dir,
        "pred": run_dir / "prediction_rows.csv.gz",
        "history": run_dir / "history.csv",
        "status": run_dir / "status.json",
    }


def audit_feature_columns(cols: list[str]) -> None:
    bad = []
    for col in cols:
        for pattern in FORBIDDEN_INPUT_PATTERNS:
            if pattern.search(col):
                bad.append(col)
                break
    if bad:
        raise RuntimeError(f"CUMULATIVE_FEATURE_LEAK: forbidden model input columns selected: {sorted(set(bad))}")


def ordered_unique(cols: list[str]) -> list[str]:
    out = []
    for col in cols:
        if col not in out:
            out.append(col)
    return out


def variant_feature_columns(variant: str) -> list[str]:
    spec = VARIANTS[variant]
    residual_mode = str(spec["residual_mode"])
    if residual_mode == "trajectory_causal":
        cols = list(BRANCH_BAND_DEEP_FEATURES)
    elif residual_mode == "window_local":
        cols = list(BRANCH_DEEP_FEATURES)
    elif residual_mode == "hybrid":
        cols = list(BRANCH_DEEP_FEATURES) + ["V_residual_low", "V_residual_low_x_T"]
    else:
        raise ValueError(f"Unknown residual_mode={residual_mode}")
    if bool(spec["cold_ema"]):
        cols += COLD_EMA_FEATURES
    cols = ordered_unique(cols)
    audit_feature_columns(cols)
    return cols


def scaling_columns(feature_cols: list[str], residual_mode: str) -> list[str]:
    cols = list(feature_cols)
    if residual_mode == "window_local":
        cols += [stat_col for _, stat_col in LOCAL_RESIDUAL_FEATURES]
    if residual_mode == "hybrid":
        cols += [stat_col for _, stat_col in HYBRID_RESIDUAL_FEATURES]
    return ordered_unique(cols)


def local_residual_features(raw_frame: pd.DataFrame, start: int, end: int) -> dict[str, np.ndarray]:
    f = raw_frame.reset_index(drop=True)
    sl = slice(start, end + 1)
    residual = (
        f["V_raw"].to_numpy(np.float64)[sl]
        - f["V_corr_raw"].to_numpy(np.float64)[sl]
    )
    low = ema_causal(residual, cutoff_hz=0.003)
    mid_lp = ema_causal(residual, cutoff_hz=0.03)
    mid = mid_lp - low
    high = residual - mid_lp
    temp = f["temperature"].to_numpy(np.float64)[sl]
    abs_i = f["absI"].to_numpy(np.float64)[sl]
    abs_di = np.abs(f["dI"].to_numpy(np.float64)[sl])
    return {
        "V_residual_raw": residual.astype(np.float32),
        "V_residual_low": low.astype(np.float32),
        "V_residual_mid": mid.astype(np.float32),
        "V_residual_high": high.astype(np.float32),
        "V_residual_low_x_T": (low * temp).astype(np.float32),
        "V_residual_mid_x_absI": (mid * abs_i).astype(np.float32),
        "V_residual_high_x_abs_dI": (high * abs_di).astype(np.float32),
    }


def ema_causal(x: np.ndarray, cutoff_hz: float, dt_sec: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    alpha = float(np.exp(-2.0 * np.pi * float(cutoff_hz) * float(dt_sec)))
    alpha = min(max(alpha, 0.0), 0.999999)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


class BranchBandsWindowDataset(Dataset):
    def __init__(
        self,
        scaled_frames: list[pd.DataFrame],
        raw_frames: list[pd.DataFrame],
        feature_cols: list[str],
        scaler,
        residual_mode: str,
        window_len: int,
        stride: int,
        sequence_target: bool,
    ):
        self.feature_cols = list(feature_cols)
        self.scaler = scaler
        self.residual_mode = str(residual_mode)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.sequence_target = bool(sequence_target)
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

    def __len__(self) -> int:
        return len(self.index)

    def _scale_feature(self, stat_col: str, values: np.ndarray) -> np.ndarray:
        if stat_col not in self.scaler.columns:
            raise KeyError(f"Scaler has no statistics for {stat_col}")
        idx = self.scaler.columns.index(stat_col)
        return ((values.astype(np.float32) - self.scaler.mean_[idx]) / self.scaler.std_[idx]).astype(np.float32)

    def _augment(self, x: np.ndarray, raw_frame: pd.DataFrame, start: int, end: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        delta = x - x[:1]
        t = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32).reshape(-1, 1)
        if not (float(t[0, 0]) == 0.0 and float(t[-1, 0]) == 1.0):
            raise AssertionError("delta_start_time must be window-local normalized position")
        parts = [x, delta, t]
        if self.residual_mode in {"window_local", "hybrid"}:
            local = local_residual_features(raw_frame, start, end)
            mapping = LOCAL_RESIDUAL_FEATURES if self.residual_mode == "window_local" else HYBRID_RESIDUAL_FEATURES
            extra = [
                self._scale_feature(stat_col, local[stat_col]).reshape(-1, 1)
                for _, stat_col in mapping
            ]
            parts.extend(extra)
        return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)

    def __getitem__(self, idx: int):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        raw = self.raw_frames[fi]
        x = self._augment(f["x"][start:end + 1], raw, start, end)
        if self.sequence_target:
            y = f["y"][start:end + 1, None].copy()
        else:
            y = np.array([f["y"][end]], dtype=np.float32)
        obs = frame_observability(raw, start, end)
        soc_val = float(f["y"][end])
        if soc_val < 0.2:
            soc_bin = 0
        elif soc_val <= 0.8:
            soc_bin = 1
        else:
            soc_bin = 2
        endpoint = raw.iloc[int(end)]
        meta = {
            "file_name": f["file_name"][end],
            "trajectory_id": f["trajectory_id"][end],
            "end_index": int(f["end_index"][end]),
            "temperature": float(f["temperature"][end]),
            "drive_cycle": f["drive_cycle"][end],
            "soc_bin": int(soc_bin),
            "endpoint_R0": float(endpoint.get("R0", np.nan)),
            "endpoint_absI": float(endpoint.get("absI", obs.get("endpoint_absI", np.nan))),
            "endpoint_V_residual_low": float(endpoint.get("V_residual_low", np.nan)),
            "endpoint_V_pol_slow_raw": float(endpoint.get("V_pol_slow_raw", np.nan)),
            **obs,
        }
        return torch.from_numpy(x), torch.from_numpy(np.asarray(y, dtype=np.float32)), meta


def endpoint_temperatures(ds: BranchBandsWindowDataset) -> np.ndarray:
    return np.asarray([ds.frames[fi]["temperature"][end] for fi, _, end in ds.index], dtype=np.float32)


def make_loader(ds: BranchBandsWindowDataset, cfg: BranchBandsImprovementConfig, *, shuffle: bool) -> DataLoader:
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
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(ds, sampler=sampler, **kwargs)


class ColdCorrectionTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        norm_kind: str,
        dropout: float,
        correction_limit: float,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.correction_limit = float(correction_limit)
        self.base = DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            norm_kind=norm_kind,
            dropout=dropout,
        )
        names = ["T", "R0", "V_pol_slow_raw", "V_residual_low", "absI"]
        missing = [name for name in names if name not in self.feature_cols]
        if missing:
            raise ValueError(f"ColdCorrectionTCN missing correction inputs: {missing}")
        self.corr_idx = [self.feature_cols.index(name) for name in names]
        hidden = max(24, hidden_size // 4)
        self.correction = nn.Sequential(
            nn.Linear(len(self.corr_idx), hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def _base_logits(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base.encode_sequence(x)
        return self.base.head(y).transpose(1, 2)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._base_logits(x)
        correction_input = x[..., self.corr_idx]
        delta_logit = self.correction_limit * torch.tanh(self.correction(correction_input))
        return torch.sigmoid(logits + delta_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def augmented_input_dim(feature_cols: list[str], residual_mode: str) -> int:
    dim = len(feature_cols) * 2 + 1
    if residual_mode == "window_local":
        dim += len(LOCAL_RESIDUAL_FEATURES)
    elif residual_mode == "hybrid":
        dim += len(HYBRID_RESIDUAL_FEATURES)
    return int(dim)


def build_model(variant: str, input_dim: int, feature_cols: list[str], cfg: BranchBandsImprovementConfig) -> nn.Module:
    spec = VARIANTS[variant]
    limit = spec["cold_head_limit"]
    if limit is None:
        return DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            kernel_size=cfg.kernel_size,
            norm_kind=cfg.norm_kind,
            dropout=cfg.dropout,
        )
    return ColdCorrectionTCN(
        input_dim=input_dim,
        feature_cols=feature_cols,
        hidden_size=cfg.hidden_size,
        layers=cfg.layers,
        kernel_size=cfg.kernel_size,
        norm_kind=cfg.norm_kind,
        dropout=cfg.dropout,
        correction_limit=float(limit),
    )


def move_float(x: torch.Tensor) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def load_fold_frames(cfg: BranchBandsImprovementConfig, fold: str, lookup: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    base_cfg = make_cfg()
    base_cfg.output_dir = cfg.base_dir
    base_cfg.base_dir = cfg.base_dir
    base_cfg = experiment_cfg(base_cfg, fold)
    feature_dir = cfg.base_dir / cfg.feature_dir
    if str(cfg.feature_dir) and feature_dir.exists():
        base_cfg.decomposed_dir = feature_dir
    configure_strict_training(base_cfg)
    base_cfg.window_len = int(cfg.window_len)
    base_cfg.stride = int(cfg.stride)
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    return add_derived_features(load_relabelled_frames(base_cfg, fold, lookup))


def train_one(variant: str, fold: str, seed: int, cfg: BranchBandsImprovementConfig, lookup: pd.DataFrame) -> dict:
    paths = run_paths(cfg, variant, fold, seed)
    if status_done(paths["status"]) and paths["pred"].exists() and not cfg.force:
        return {"status": "cached", "variant": variant, "fold_name": fold, "seed": seed}
    paths["dir"].mkdir(parents=True, exist_ok=True)
    started = time.time()
    set_seed(seed)
    spec = VARIANTS[variant]
    residual_mode = str(spec["residual_mode"])
    raw_frames = load_fold_frames(cfg, fold, lookup)
    feature_cols = variant_feature_columns(variant)
    scale_cols = scaling_columns(feature_cols, residual_mode)
    available = set().union(*(set(f.columns) for split in raw_frames.values() for f in split))
    missing = [c for c in scale_cols if c not in available]
    if missing:
        raise KeyError(f"{variant} {fold}: missing columns {missing}")
    scaled, scaler = make_scaled_frames_for_ablation(raw_frames, scale_cols)
    train_ds = BranchBandsWindowDataset(
        scaled["train"],
        raw_frames["train"],
        feature_cols,
        scaler,
        residual_mode,
        cfg.window_len,
        cfg.stride,
        sequence_target=True,
    )
    test_ds = BranchBandsWindowDataset(
        scaled["test"],
        raw_frames["test"],
        feature_cols,
        scaler,
        residual_mode,
        cfg.window_len,
        1,
        sequence_target=False,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"{variant} {fold}: empty dataset train={len(train_ds)} test={len(test_ds)}")
    input_dim = augmented_input_dim(feature_cols, residual_mode)
    model = build_model(variant, input_dim, feature_cols, cfg).to(device)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    test_loader = make_loader(test_ds, cfg, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        means = []
        rexes = []
        smooths = []
        group_losses = {}
        for x, y, meta in train_loader:
            x = move_float(x)
            y = move_float(y)
            pred = model.forward_sequence(x)
            loss, mean_loss, rex_loss, smooth_loss, by_group = temp_balanced_rex_loss(
                pred,
                y,
                meta,
                cfg.lambda_rex,
                cfg.rex_group,
                cfg.loss_kind,
                cfg.huber_beta,
                cfg.lambda_smooth,
                cfg.endpoint_loss_weight,
                cfg.lambda_worst,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            means.append(float(mean_loss.detach().cpu()))
            rexes.append(float(rex_loss.detach().cpu()))
            smooths.append(float(smooth_loss.detach().cpu()))
            for key, value in by_group.items():
                group_losses.setdefault(str(key), []).append(float(value))
        row = {
            "variant": variant,
            "fold_name": fold,
            "seed": int(seed),
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "mean_huber": float(np.mean(means)),
            "rex_var": float(np.mean(rexes)),
            "smooth_delta_loss": float(np.mean(smooths)),
            "residual_mode": residual_mode,
            "cold_ema": bool(spec["cold_ema"]),
            "cold_head_limit": spec["cold_head_limit"],
        }
        for key, values in group_losses.items():
            safe = key.replace(".", "p").replace("-", "N").replace("/", "_")
            row[f"train_loss_group_{safe}"] = float(np.mean(values))
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(
                f"{variant} {fold} seed={seed} ep={ep} loss={row['train_loss']:.5f} "
                f"mean={row['mean_huber']:.5f} rex={row['rex_var']:.6f}",
                flush=True,
            )
    pred = predict_model(model, test_loader, variant, fold, seed)
    pred["residual_mode"] = residual_mode
    pred["cold_ema"] = bool(spec["cold_ema"])
    pred["cold_head_limit"] = spec["cold_head_limit"]
    pred.to_csv(paths["pred"], index=False, compression="gzip")
    pd.DataFrame(history).to_csv(paths["history"], index=False)
    write_json(
        paths["status"],
        {
            "status": "done",
            "variant": variant,
            "fold_name": fold,
            "seed": int(seed),
            "duration_s": time.time() - started,
            "train_windows": len(train_ds),
            "test_windows": len(test_ds),
            "feature_cols": feature_cols,
            "scaling_cols": scale_cols,
            "input_dim": input_dim,
            "residual_mode": residual_mode,
            "cold_ema": bool(spec["cold_ema"]),
            "cold_head_limit": spec["cold_head_limit"],
            "strict_no_soc_input": True,
            "strict_no_cumulative_input": True,
            "explicit_current_integration_state_update": False,
            "current_policy": "instantaneous excitation only; no integrated current state update",
            "delta_start_time_policy": "window-local normalized t in [0, 1], no absolute time or trajectory progress input",
        },
    )
    return {"status": "done", "variant": variant, "fold_name": fold, "seed": int(seed)}


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, variant: str, fold: str, seed: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        pred = model(move_float(x)).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["model_name"] = variant
        mdf["run_model_name"] = variant
        mdf["target_label"] = "physical"
        mdf["label_type"] = "physical_smoothQ"
        mdf["fold_name"] = fold
        mdf["experiment"] = fold
        mdf["seed"] = int(seed)
        mdf["y_true"] = yy
        mdf["y_pred"] = pred
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["temperature_C"] = out["temperature"].astype(float)
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        out["is_plateau_20_80"] = (out["y_true"] >= 0.2) & (out["y_true"] <= 0.8)
        out["label_policy"] = "physical_smoothQ"
    return out


def load_completed_predictions(cfg: BranchBandsImprovementConfig) -> pd.DataFrame:
    chunks = []
    for variant in cfg.variants:
        for fold in cfg.folds:
            for seed in cfg.seeds:
                paths = run_paths(cfg, variant, fold, int(seed))
                if status_done(paths["status"]) and paths["pred"].exists():
                    chunks.append(pd.read_csv(paths["pred"]))
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()


def fold_target_type(fold: str) -> str:
    if fold == "Exp D":
        return "included_diagnostic"
    if fold in {"Omit N10", "Omit 50"}:
        return "outside"
    return "omitted"


def compute_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows = []
    temp_rows = []
    obs_rows = []
    obs_vs_rows = []
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for (model, seed, fold), g in pred.groupby(["model_name", "seed", "fold_name"]):
        target_temp = float(EXPERIMENTS[fold]["omitted_temp_C"])
        target = g[np.isclose(g["temperature_C"].astype(float), target_temp)]
        if len(target):
            row = {
                "model_name": model,
                "seed": int(seed),
                "fold_name": fold,
                "target_temperature_C": target_temp,
                "target_type": fold_target_type(fold),
                "source": "branchbands_improvement",
            }
            row.update(metrics_for_group(target))
            result_rows.append(row)
        for temp, tg in g.groupby("temperature_C"):
            row = {
                "model_name": model,
                "seed": int(seed),
                "fold_name": fold,
                "temperature_C": float(temp),
                "source": "branchbands_improvement",
            }
            row.update(metrics_for_group(tg))
            temp_rows.append(row)
        if "branch_band_energy" in g.columns:
            gg = g.copy()
            try:
                gg["observability_bin"] = pd.qcut(
                    gg["branch_band_energy"].rank(method="first"),
                    q=3,
                    labels=["low", "mid", "high"],
                )
            except ValueError:
                gg["observability_bin"] = "all"
            corr = np.nan
            if gg["branch_band_energy"].notna().any() and gg["abs_error"].notna().any():
                corr = gg[["branch_band_energy", "abs_error"]].corr(method="spearman").iloc[0, 1]
            for obs_bin, og in gg.groupby("observability_bin", observed=False):
                if len(og):
                    row = {
                        "model_name": model,
                        "seed": int(seed),
                        "fold_name": fold,
                        "observability_bin": str(obs_bin),
                        "mean_branch_band_energy": float(og["branch_band_energy"].mean()),
                        "source": "branchbands_improvement",
                    }
                    row.update(metrics_for_group(og))
                    obs_rows.append(row)
                    obs_vs_rows.append({
                        "model_name": model,
                        "seed": int(seed),
                        "fold_name": fold,
                        "observability_feature": "branch_band_energy",
                        "observability_bin": str(obs_bin),
                        "feature_mean": float(og["branch_band_energy"].mean()),
                        "MAE_pct": float(og["abs_error"].mean() * 100.0),
                        "RMSE_pct": float(np.sqrt(np.mean(og["error"].to_numpy(float) ** 2)) * 100.0),
                        "n_samples": int(len(og)),
                        "spearman_observability_vs_abs_error": float(corr) if pd.notna(corr) else np.nan,
                        "source": "branchbands_improvement",
                    })
    results = pd.DataFrame(result_rows)
    by_temp = pd.DataFrame(temp_rows)
    focus = summarize_focus(results) if len(results) else pd.DataFrame()
    return results, by_temp, focus, pd.DataFrame(obs_rows), pd.DataFrame(obs_vs_rows)


def build_promotion_table(results: pd.DataFrame, focus: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    base_name = "BandTCN_w150_base"
    base = results[results["model_name"].eq(base_name)]
    if base.empty:
        return pd.DataFrame()
    base_by_fold = {row["fold_name"]: float(row["MAE_pct"]) for _, row in base.iterrows()}
    base_exp_d = base_by_fold.get("Exp D", np.nan)
    for model, g in results.groupby("model_name"):
        by_fold = {row["fold_name"]: float(row["MAE_pct"]) for _, row in g.iterrows()}
        outside_vals = [by_fold.get("Omit N10", np.nan), by_fold.get("Omit 50", np.nan)]
        outside_vals = [v for v in outside_vals if pd.notna(v)]
        outside_avg = float(np.mean(outside_vals)) if outside_vals else np.nan
        outside_worst = float(np.max(outside_vals)) if outside_vals else np.nan
        omit_n10_improvement = base_by_fold.get("Omit N10", np.nan) - by_fold.get("Omit N10", np.nan)
        omit_50_improvement = base_by_fold.get("Omit 50", np.nan) - by_fold.get("Omit 50", np.nan)
        exp_d_degrade = by_fold.get("Exp D", np.nan) - base_exp_d
        outside_rule = (
            (pd.notna(outside_avg) and outside_avg < 4.5)
            or (pd.notna(outside_worst) and outside_worst < 6.0)
            or (pd.notna(omit_n10_improvement) and omit_n10_improvement > 0.5)
            or (pd.notna(omit_50_improvement) and omit_50_improvement > 0.5)
        )
        expd_rule = pd.notna(exp_d_degrade) and exp_d_degrade <= 0.2
        rows.append({
            "model_name": model,
            "outside_avg_MAE_pct": outside_avg,
            "outside_worst_MAE_pct": outside_worst,
            "omit_N10_improvement_pctp_vs_base": omit_n10_improvement,
            "omit_50_improvement_pctp_vs_base": omit_50_improvement,
            "ExpD_degradation_pctp_vs_base": exp_d_degrade,
            "outside_rule_pass": bool(outside_rule),
            "ExpD_rule_pass": bool(expd_rule),
            "promote_to_3seeds": bool(outside_rule and expd_rule and model != base_name),
        })
    return pd.DataFrame(rows)


def build_cold_fusion(pred: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pred.empty or results.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_name = "BandTCN_w150_base"
    cold_names = [m for m in pred["model_name"].unique() if "coldEMA" in str(m)]
    if base_name not in set(pred["model_name"]) or not cold_names:
        return pd.DataFrame(), pd.DataFrame()
    base_focus = summarize_focus(results[results["model_name"].eq(base_name)])
    all_base = float(base_focus.loc[base_focus["scope"].eq("all_target_folds"), "average_MAE_pct"].iloc[0]) if len(base_focus) and any(base_focus["scope"].eq("all_target_folds")) else np.nan
    base_outside = results[(results["model_name"].eq(base_name)) & (results["fold_name"].isin(["Omit N10", "Omit 50"]))]["MAE_pct"].mean()
    candidate = None
    best_outside_gain = 0.0
    for name in cold_names:
        rg = results[results["model_name"].eq(name)]
        if rg.empty:
            continue
        cold_outside = rg[rg["fold_name"].isin(["Omit N10", "Omit 50"])]["MAE_pct"].mean()
        cold_focus = summarize_focus(rg)
        cold_all = float(cold_focus.loc[cold_focus["scope"].eq("all_target_folds"), "average_MAE_pct"].iloc[0]) if len(cold_focus) and any(cold_focus["scope"].eq("all_target_folds")) else np.nan
        outside_gain = float(base_outside - cold_outside) if pd.notna(cold_outside) and pd.notna(base_outside) else 0.0
        hurts_average = pd.notna(cold_all) and pd.notna(all_base) and cold_all > all_base
        if outside_gain > best_outside_gain and hurts_average:
            candidate = name
            best_outside_gain = outside_gain
    if candidate is None:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["fold_name", "seed", "trajectory_id", "end_index"]
    rows = []
    weight_rows = []
    for (fold, seed), fg in pred[pred["model_name"].isin([base_name, candidate])].groupby(["fold_name", "seed"]):
        base = fg[fg["model_name"].eq(base_name)].copy()
        cold = fg[fg["model_name"].eq(candidate)][keys + ["y_pred"]].rename(columns={"y_pred": "p_cold"})
        m = base.merge(cold, on=keys, how="inner")
        if m.empty:
            continue
        p_branch = m["y_pred"].to_numpy(float)
        p_cold = m["p_cold"].to_numpy(float)
        disagreement = np.abs(p_branch - p_cold)
        temp = m["temperature_C"].astype(float).to_numpy()
        cold_gate = 1.0 / (1.0 + np.exp((temp - 5.0) / 4.0))
        high_gate = 1.0 / (1.0 + np.exp(-(temp - 42.0) / 4.0))
        disagreement_n = disagreement / (np.nanpercentile(disagreement, 90) + 1e-12)
        band_energy = m.get("branch_band_energy", pd.Series(0.0, index=m.index)).astype(float).to_numpy()
        band_n = band_energy / (np.nanpercentile(band_energy, 90) + 1e-12)
        w_cold = np.clip(0.12 + 0.35 * cold_gate + 0.15 * high_gate + 0.15 * disagreement_n + 0.08 * band_n, 0.05, 0.75)
        w_branch = 1.0 - w_cold
        y_pred = w_branch * p_branch + w_cold * p_cold
        out = m.copy()
        out["model_name"] = "BranchBands_FilteredCold_fusion"
        out["run_model_name"] = out["model_name"]
        out["fusion_cold_expert"] = candidate
        out["y_pred"] = y_pred
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        rows.append(out)
        wr = out[keys + ["temperature_C"]].copy()
        wr["fusion_cold_expert"] = candidate
        wr["w_branch"] = w_branch
        wr["w_cold"] = w_cold
        wr["expert_disagreement"] = disagreement
        wr["T"] = m["temperature_C"].astype(float).to_numpy()
        wr["R0"] = m.get("endpoint_R0", pd.Series(np.nan, index=m.index)).astype(float).to_numpy()
        wr["V_residual_low"] = m.get("endpoint_V_residual_low", pd.Series(np.nan, index=m.index)).astype(float).to_numpy()
        wr["V_pol_slow_raw"] = m.get("endpoint_V_pol_slow_raw", pd.Series(np.nan, index=m.index)).astype(float).to_numpy()
        wr["absI"] = m.get("endpoint_absI", pd.Series(np.nan, index=m.index)).astype(float).to_numpy()
        wr["dI_energy"] = m.get("dI_energy", pd.Series(np.nan, index=m.index)).astype(float).to_numpy()
        wr["branch_band_energy"] = band_energy
        weight_rows.append(wr)
    return (
        pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(),
        pd.concat(weight_rows, ignore_index=True, sort=False) if weight_rows else pd.DataFrame(),
    )


def build_decision_summary(results: pd.DataFrame, focus: pd.DataFrame, promotion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_name = "BandTCN_w150_base"

    def add_row(question: str, status: str, answer: str, evidence: str = ""):
        rows.append({
            "question": question,
            "status": status,
            "answer": answer,
            "evidence": evidence,
        })

    def fold_answer(fold: str, label: str):
        if results.empty:
            add_row(label, "INCOMPLETE", "No target-fold result rows are available.")
            return
        g = results[results["fold_name"].eq(fold)].copy()
        base = g[g["model_name"].eq(base_name)]
        if g.empty or base.empty:
            add_row(label, "INCOMPLETE", f"{fold} results are not complete enough to compare against base.")
            return
        g = g.sort_values("MAE_pct")
        best = g.iloc[0]
        base_mae = float(base["MAE_pct"].iloc[0])
        improvement = base_mae - float(best["MAE_pct"])
        status = "PASS" if improvement > 0 else "NO_IMPROVEMENT"
        add_row(
            label,
            status,
            f"Best is {best['model_name']} at {float(best['MAE_pct']):.4f}% MAE; base is {base_mae:.4f}% MAE; improvement is {improvement:.4f} percentage points.",
            g[["model_name", "MAE_pct", "RMSE_pct", "jitter_ratio", "catastrophic_error_rate_5pct"]].to_json(orient="records"),
        )

    fold_answer("Omit N10", "Which variant improves -10 degC?")
    fold_answer("Omit 50", "Which variant improves 50 degC?")

    if promotion.empty:
        add_row("Does outside robustness improve?", "INCOMPLETE", "Promotion/outside table is unavailable.")
        add_row("Is Exp D strong performance preserved?", "INCOMPLETE", "Promotion/outside table is unavailable.")
        add_row("Which variants promote to 3 seeds?", "INCOMPLETE", "Promotion/outside table is unavailable.")
    else:
        candidates = promotion[promotion["model_name"].ne(base_name)].copy()
        outside_cols = ["outside_avg_MAE_pct", "outside_worst_MAE_pct"]
        if candidates[outside_cols].notna().all(axis=1).any():
            valid = candidates[candidates[outside_cols].notna().all(axis=1)].sort_values("outside_avg_MAE_pct")
            best = valid.iloc[0]
            base = promotion[promotion["model_name"].eq(base_name)]
            base_outside = float(base["outside_avg_MAE_pct"].iloc[0]) if len(base) and pd.notna(base["outside_avg_MAE_pct"].iloc[0]) else np.nan
            gain = base_outside - float(best["outside_avg_MAE_pct"]) if pd.notna(base_outside) else np.nan
            add_row(
                "Does outside robustness improve?",
                "PASS" if pd.notna(gain) and gain > 0 else "NO_IMPROVEMENT",
                f"Best outside avg is {best['model_name']} at {float(best['outside_avg_MAE_pct']):.4f}% MAE; base outside avg is {base_outside:.4f}% MAE; gain is {gain:.4f} percentage points.",
                valid[["model_name", "outside_avg_MAE_pct", "outside_worst_MAE_pct", "omit_N10_improvement_pctp_vs_base", "omit_50_improvement_pctp_vs_base"]].to_json(orient="records"),
            )
        else:
            add_row("Does outside robustness improve?", "INCOMPLETE", "Omit N10 and Omit 50 are both required before outside robustness can be judged.")

        expd_valid = candidates[candidates["ExpD_degradation_pctp_vs_base"].notna()].copy()
        if len(expd_valid):
            preserved = expd_valid[expd_valid["ExpD_degradation_pctp_vs_base"] <= 0.2]
            add_row(
                "Is Exp D strong performance preserved?",
                "PASS" if len(preserved) else "FAIL",
                f"{len(preserved)}/{len(expd_valid)} non-base variants keep Exp D degradation <= 0.2 percentage points.",
                expd_valid[["model_name", "ExpD_degradation_pctp_vs_base", "ExpD_rule_pass"]].to_json(orient="records"),
            )
        else:
            add_row("Is Exp D strong performance preserved?", "INCOMPLETE", "Exp D rows are required before preservation can be judged.")

        promoted = candidates[candidates["promote_to_3seeds"].fillna(False)].copy()
        add_row(
            "Which variants promote to 3 seeds?",
            "PASS" if len(promoted) else ("NO_PROMOTION" if candidates["outside_avg_MAE_pct"].notna().any() else "INCOMPLETE"),
            ", ".join(promoted["model_name"].astype(str).tolist()) if len(promoted) else "No promoted variant yet.",
            promoted.to_json(orient="records") if len(promoted) else "",
        )

    full_scopes = focus[focus["scope"].eq("all_target_folds")] if len(focus) else pd.DataFrame()
    full_complete = len(full_scopes) >= len(VARIANTS)
    if full_complete:
        best = full_scopes.sort_values("average_MAE_pct").iloc[0]
        add_row(
            "Best all-target 1-seed model",
            "PASS",
            f"{best['model_name']} has the best all-target average MAE at {float(best['average_MAE_pct']):.4f}%.",
            full_scopes[["model_name", "average_MAE_pct", "worst_MAE_pct", "average_RMSE_pct", "worst_RMSE_pct"]].to_json(orient="records"),
        )
    else:
        add_row("Best all-target 1-seed model", "INCOMPLETE", "All 7 variants need full target-fold coverage before this is meaningful.")

    if len(promotion) and promotion["promote_to_3seeds"].fillna(False).any():
        add_row(
            "Is NoCC viable as main model?",
            "INCOMPLETE",
            "A one-seed promotion candidate exists, but main-model viability still requires 3-seed confirmation and comparison to leak-free baselines.",
        )
    else:
        add_row(
            "Is NoCC viable as main model?",
            "NO",
            "At this stage it should remain an ablation, not a main-model claim.",
        )
    add_row(
        "Safe paper wording",
        "PASS",
        "Current is used only as instantaneous excitation, not integrated into SOC state. Do not claim NoCC proves current integration unnecessary or solves temperature extrapolation.",
    )
    return pd.DataFrame(rows)


def write_audit_and_schema(cfg: BranchBandsImprovementConfig) -> None:
    rows = []
    schema_rows = []
    model_sources = "\n".join(inspect.getsource(obj) for obj in [DeepNoLeakTCN, ColdCorrectionTCN])
    explicit_patterns = [
        r"Q_eff",
        r"dt\s*/\s*3600",
        r"soc\s*=.*[-+].*\bI\b",
        r"current_integrat",
        r"coulomb",
    ]
    explicit_hits = [pat for pat in explicit_patterns if re.search(pat, model_sources, flags=re.I)]
    for variant in cfg.variants:
        spec = VARIANTS[variant]
        feature_cols = variant_feature_columns(variant)
        residual_mode = str(spec["residual_mode"])
        input_dim = augmented_input_dim(feature_cols, residual_mode)
        rows.append({
            "audit_item": "forbidden_input_feature_scan",
            "variant": variant,
            "status": "PASS",
            "evidence": ";".join(feature_cols),
        })
        for pos, col in enumerate(feature_cols):
            schema_rows.append({
                "variant": variant,
                "input_order": pos,
                "input_name": col,
                "input_stage": "scaled_base_feature",
                "source": feature_source(col),
                "residual_mode": residual_mode,
                "cold_ema": bool(spec["cold_ema"]),
                "cold_head_limit": spec["cold_head_limit"],
                "forbidden_token_hit": False,
            })
        for pos, col in enumerate(feature_cols):
            schema_rows.append({
                "variant": variant,
                "input_order": len(feature_cols) + pos,
                "input_name": f"delta_start({col})",
                "input_stage": "window_delta_from_first_sample",
                "source": "x_t minus x_window_start; no prior window SOC or absolute time",
                "residual_mode": residual_mode,
                "cold_ema": bool(spec["cold_ema"]),
                "cold_head_limit": spec["cold_head_limit"],
                "forbidden_token_hit": False,
            })
        schema_rows.append({
            "variant": variant,
            "input_order": len(feature_cols) * 2,
            "input_name": "delta_start_time",
            "input_stage": "window_local_position",
            "source": "linspace 0..1 inside current window only",
            "residual_mode": residual_mode,
            "cold_ema": bool(spec["cold_ema"]),
            "cold_head_limit": spec["cold_head_limit"],
            "forbidden_token_hit": False,
        })
        extra = []
        if residual_mode == "window_local":
            extra = LOCAL_RESIDUAL_FEATURES
        elif residual_mode == "hybrid":
            extra = HYBRID_RESIDUAL_FEATURES
        for extra_pos, (name, stat_col) in enumerate(extra):
            schema_rows.append({
                "variant": variant,
                "input_order": len(feature_cols) * 2 + 1 + extra_pos,
                "input_name": name,
                "input_stage": "window_local_residual_band",
                "source": f"causal within-window residual feature, standardized with train statistics of {stat_col}",
                "residual_mode": residual_mode,
                "cold_ema": bool(spec["cold_ema"]),
                "cold_head_limit": spec["cold_head_limit"],
                "forbidden_token_hit": False,
            })
        rows.append({
            "audit_item": "delta_start_time_assert",
            "variant": variant,
            "status": "PASS",
            "evidence": "position channel is generated as linspace(0,1) after each window is sliced; no end_index, file_name, trajectory_id, time_index, or progress is appended",
        })
        rows.append({
            "audit_item": "input_dim",
            "variant": variant,
            "status": "PASS",
            "evidence": str(input_dim),
        })
    rows.append({
        "audit_item": "explicit_current_integration_state_update_scan",
        "variant": "all",
        "status": "PASS" if not explicit_hits else "FAIL",
        "evidence": "no matching model-source pattern" if not explicit_hits else ";".join(explicit_hits),
    })
    rows.append({
        "audit_item": "branch_bands_feature_source",
        "variant": "all",
        "status": "PASS",
        "evidence": "branch bands use V_pol fast/mid/slow and V_raw minus V_corr residual bands; trajectory mode uses causal EMA over each trajectory, local modes restart causal EMA at the current window start",
    })
    audit = pd.DataFrame(rows)
    if audit["status"].eq("FAIL").any():
        audit.to_csv(cfg.base_dir / "branchbands_leakage_audit.csv", index=False)
        raise RuntimeError("EXPLICIT_INTEGRATION_LEAK")
    audit.to_csv(cfg.base_dir / "branchbands_leakage_audit.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(cfg.base_dir / "branchbands_input_schema.csv", index=False)


def feature_source(col: str) -> str:
    if col in COLD_EMA_FEATURES:
        return "trajectory-causal EMA or deviation feature computed before scaling; no future values"
    if col.startswith("V_residual"):
        return "V_raw - V_corr derived residual band; causal low/mid/high filters"
    if col.startswith("V_pol_") and col.endswith("_raw"):
        return "polarization branch feature from voltage decomposition"
    if "_x_" in col:
        return "instantaneous interaction term from label-free voltage/current/temperature features"
    return "raw or decomposed instantaneous feature"


def write_reports(
    cfg: BranchBandsImprovementConfig,
    results: pd.DataFrame,
    by_temp: pd.DataFrame,
    focus: pd.DataFrame,
    promotion: pd.DataFrame,
    decision_summary: pd.DataFrame,
    fusion_results: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# BranchBands TCN Strict NoCC Improvement Report",
        "",
        "## Strict policy",
        "- SOC input, window-start SOC, initial SOC label, SOC_CC, cumulative Ah, absolute time, trajectory progress, and explicit integrated-current SOC state updates are not used.",
        "- Current is used only as instantaneous excitation, not integrated into SOC state.",
        "- `delta_start_time` is the normalized position inside the current window.",
        "- Label policy is `physical_smoothQ`.",
        "",
    ]
    if len(decision_summary):
        lines += ["## Decision summary", table(decision_summary), ""]
    if len(focus):
        lines += ["## Focus summary", table(focus.sort_values(["scope", "average_MAE_pct"])), ""]
    if len(promotion):
        lines += ["## Promotion screen", table(promotion.sort_values("outside_avg_MAE_pct")), ""]
    if len(results):
        cold = results[results["fold_name"].eq("Omit N10")].sort_values("MAE_pct")
        hot = results[results["fold_name"].eq("Omit 50")].sort_values("MAE_pct")
        expd = results[results["fold_name"].eq("Exp D")].sort_values("MAE_pct")
        if len(cold):
            lines += ["## Omit -10 degC ranking", table(cold[["model_name", "MAE_pct", "RMSE_pct", "jitter_ratio", "catastrophic_error_rate_5pct"]]), ""]
        if len(hot):
            lines += ["## Omit 50 degC ranking", table(hot[["model_name", "MAE_pct", "RMSE_pct", "jitter_ratio", "catastrophic_error_rate_5pct"]]), ""]
        if len(expd):
            lines += ["## Exp D preservation", table(expd[["model_name", "MAE_pct", "RMSE_pct", "jitter_ratio", "catastrophic_error_rate_5pct"]]), ""]
    if len(fusion_results):
        lines += ["## Optional fusion", "FilteredCold fusion was triggered because a cold variant improved outside robustness while hurting all-fold average.", ""]
    else:
        lines += ["## Optional fusion", "FilteredCold fusion was not triggered by the screening rule.", ""]
    lines += [
        "## Required interpretation",
        "- If a variant improves -10 degC, treat it as better cold robustness within this one-seed screen only.",
        "- If a variant improves 50 degC, treat it as better hot-side outside robustness within this one-seed screen only.",
        "- Exp D is considered preserved only when its MAE degradation versus base is no more than 0.2 percentage points.",
        "- This remains a strict NoCC ablation unless outside/cold performance is consistently promoted and confirmed over three seeds.",
        "- Do not claim NoCC proves current integration unnecessary or solves temperature extrapolation.",
    ]
    (cfg.base_dir / "branchbands_improvement_report.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_and_write(cfg: BranchBandsImprovementConfig):
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    write_audit_and_schema(cfg)
    pred = load_completed_predictions(cfg)
    results, by_temp, focus, obs, obs_vs = compute_tables(pred)
    results.to_csv(cfg.base_dir / "branchbands_improvement_results.csv", index=False)
    by_temp.to_csv(cfg.base_dir / "branchbands_improvement_by_temperature.csv", index=False)
    focus.to_csv(cfg.base_dir / "branchbands_improvement_focus.csv", index=False)
    obs.to_csv(cfg.base_dir / "branchbands_observability_metrics.csv", index=False)
    obs_vs.to_csv(cfg.base_dir / "branchbands_observability_vs_error.csv", index=False)
    residual_models = [
        "BandTCN_w150_base",
        "BandTCN_w150_residual_windowLocal",
        "BandTCN_w150_residual_hybrid",
    ]
    residual_results = results[results["model_name"].isin(residual_models)].copy() if len(results) else pd.DataFrame()
    residual_results.to_csv(cfg.base_dir / "branchband_residual_mode_results.csv", index=False)
    promotion = build_promotion_table(results, focus)
    promotion.to_csv(cfg.base_dir / "branchbands_improvement_promotion.csv", index=False)
    decision_summary = build_decision_summary(results, focus, promotion)
    decision_summary.to_csv(cfg.base_dir / "branchbands_improvement_decision_summary.csv", index=False)
    fusion_pred, fusion_weights = build_cold_fusion(pred, results)
    if len(fusion_pred):
        fusion_results, fusion_by_temp, fusion_focus, _, _ = compute_tables(fusion_pred)
    else:
        fusion_results, fusion_by_temp, fusion_focus = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    fusion_results.to_csv(cfg.base_dir / "branchbands_cold_expert_fusion_results.csv", index=False)
    fusion_by_temp.to_csv(cfg.base_dir / "branchbands_cold_expert_fusion_by_temperature.csv", index=False)
    fusion_focus.to_csv(cfg.base_dir / "branchbands_cold_expert_fusion_focus.csv", index=False)
    fusion_weights.to_csv(cfg.base_dir / "branchbands_cold_expert_weights.csv", index=False)
    if cfg.save_predictions and len(pred):
        pred.to_csv(cfg.base_dir / "branchbands_improvement_prediction_rows.csv.gz", index=False, compression="gzip")
    write_reports(cfg, results, by_temp, focus, promotion, decision_summary, fusion_results)
    metadata = {
        **asdict(cfg),
        "variants": list(cfg.variants),
        "folds": list(cfg.folds),
        "strict_no_soc_input": True,
        "strict_no_cumulative_input": True,
        "explicit_current_integration_state_update": False,
        "amp_used": False,
        "device": str(device),
    }
    write_json(cfg.base_dir / "branchbands_improvement_metadata.json", metadata)
    return pred, results, by_temp, focus


def run_all(cfg: BranchBandsImprovementConfig):
    cfg.base_dir = Path(cfg.base_dir)
    configure_torch_runtime()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    write_audit_and_schema(cfg)
    lookup = load_smoothq_lookup(cfg.base_dir)
    statuses = []
    for seed in cfg.seeds:
        for fold in cfg.folds:
            for variant in cfg.variants:
                st = train_one(str(variant), str(fold), int(seed), cfg, lookup)
                statuses.append(st)
                pd.DataFrame(statuses).to_csv(cfg.base_dir / "branchbands_improvement_run_status_log.csv", index=False)
                aggregate_and_write(cfg)
    return aggregate_and_write(cfg)


def parse_args():
    p = argparse.ArgumentParser(description="Strict NoCC BranchBands TCN outside/cold robustness screening.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--output-prefix", default="branchbands_improvement")
    p.add_argument("--folds", nargs="+", default=list(FOLDS), choices=list(FOLDS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--feature-dir", default="decomposed_features_train_temp_minus10_0_10_20_25_50")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--force", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BranchBandsImprovementConfig(
        base_dir=args.base_dir,
        output_prefix=args.output_prefix,
        folds=tuple(args.folds),
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
        feature_dir=args.feature_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        stride=int(args.stride),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        force=bool(args.force),
    )
    if args.audit_only:
        cfg.base_dir.mkdir(parents=True, exist_ok=True)
        write_audit_and_schema(cfg)
        return
    if args.aggregate_only:
        aggregate_and_write(cfg)
        return
    run_all(cfg)


if __name__ == "__main__":
    main()
