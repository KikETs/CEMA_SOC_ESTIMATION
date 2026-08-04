from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import inspect
import json
import math
import random
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import make_cfg
from .deep_no_leak_experiment import (
    CausalConvBlock,
    augment_window_tensor,
    augmented_input_dim,
)
from .models import DecomposedWindowDataset, collate_meta_to_frame
from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .training import make_scaled_frames_for_ablation


NO_CC_FOLDS = ("Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50")
REFERENCE_MODELS = (
    "A3_V_raw_I_T",
    "R5_GATED_AUG_REX",
    "NeuralECM_Tamb_Tcore",
    "ThermoGuardSOC_rule_soft_high",
)
FORBIDDEN_FEATURE_PATTERNS = (
    re.compile(r"SOC_CC", re.I),
    re.compile(r"SOC_usable", re.I),
    re.compile(r"SOC_physical", re.I),
    re.compile(r"cumulative", re.I),
    re.compile(r"q_cutoff", re.I),
    re.compile(r"Q_ref", re.I),
    re.compile(r"\bAh\b", re.I),
)

VCORR_FEATURES = [
    "V_corr_raw",
    "T",
    "T_eff_proxy",
    "V_corr_ema50",
    "V_corr_ema200",
    "V_corr_delta",
    "V_corr_abs_slope",
]
DYNAMIC_FEATURES = [
    "V_raw",
    "I_raw",
    "T",
    "V_corr_raw",
    "V_pol_raw",
    "V_hys_raw",
    "V_ohm_raw",
    "R0",
    "dI",
    "absI",
    "T_eff_proxy",
    "heat_proxy_no_cc",
    "excitation_score",
    "V_response_amp",
]
THERMAL_FEATURES = [
    "T",
    "T_eff_proxy",
    "heat_proxy_no_cc",
    "absI",
    "R0",
    "V_corr_raw",
    "V_corr_ema200",
]
SURFACE_FEATURES = [
    "V_corr_raw",
    "T",
    "T_eff_proxy",
    "R0",
    "absI",
    "V_ohm_raw",
    "V_response_amp",
]


def ordered_union(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for col in group:
            if col not in out:
                out.append(col)
    return out


@dataclass
class NoCCConfig:
    base_dir: Path = Path(".")
    output_prefix: str = "no_cc"
    folds: tuple[str, ...] = NO_CC_FOLDS
    variants: tuple[str, ...] = (
        "NoCC_VcorrInverse",
        "NoCC_DynamicResponse",
        "NoCC_BoundedCorrection_0p03",
        "NoCC_BoundedCorrection_0p05",
        "NoCC_BoundedCorrection_0p08",
        "NoCC_ThermoGuard",
    )
    seed: int = 0
    encoder: str = "tcn"
    window_len: int = 50
    stride: int = 1
    epochs: int = 80
    batch_size: int = 4096
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 96
    layers: int = 5
    kernel_size: int = 5
    dropout: float = 0.06
    norm_kind: str = "channel"
    lambda_rex: float = 0.30
    lambda_worst: float = 0.0
    huber_beta: float = 0.02
    window_feature_mode: str = "delta_start_time"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 10
    patience: int = 20
    min_delta: float = 2e-5
    warmup_epochs: int = 25
    save_predictions: bool = True


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ema_causal(x: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    alpha = float(math.exp(-1.0 / max(float(tau), 1e-6)))
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y.astype(np.float32)


def add_no_cc_derived_features(frames: dict[str, list[pd.DataFrame]]) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = {}
    for split, split_frames in frames.items():
        out[split] = []
        for frame in split_frames:
            f = frame.copy()
            v_corr = f["V_corr_raw"].to_numpy(np.float64)
            v_raw = f["V_raw"].to_numpy(np.float64)
            i = f["I_raw"].to_numpy(np.float64)
            abs_i = np.abs(i)
            di = f["dI"].to_numpy(np.float64) if "dI" in f.columns else np.r_[0.0, np.diff(i)]
            r0 = np.abs(f["R0"].to_numpy(np.float64)) if "R0" in f.columns else np.zeros_like(i)
            heat = ema_causal((abs_i ** 2) * np.maximum(r0, 1e-6), tau=300.0)
            vc50 = ema_causal(v_corr, tau=50.0)
            vc200 = ema_causal(v_corr, tau=200.0)
            v_delta = np.r_[0.0, np.diff(v_corr)].astype(np.float32)
            d_i_energy = ema_causal(di ** 2, tau=20.0)
            abs_i_energy = ema_causal(abs_i ** 2, tau=20.0)
            f["heat_proxy_no_cc"] = heat
            f["T_eff_proxy"] = f["T"].to_numpy(np.float32) + (4.0 * np.tanh(20.0 * heat)).astype(np.float32)
            f["V_corr_ema50"] = vc50
            f["V_corr_ema200"] = vc200
            f["V_corr_delta"] = v_delta
            f["V_corr_abs_slope"] = np.abs(v_delta).astype(np.float32)
            f["excitation_score"] = np.sqrt(d_i_energy + 0.25 * abs_i_energy).astype(np.float32)
            f["V_response_amp"] = (
                ema_causal(np.abs(v_raw - v_corr), tau=50.0) + np.abs(v_delta)
            ).astype(np.float32)
            out[split].append(f)
    return out


def audit_feature_columns(cols: list[str]):
    bad = []
    for col in cols:
        for pat in FORBIDDEN_FEATURE_PATTERNS:
            if pat.search(col):
                bad.append(col)
                break
    if bad:
        raise RuntimeError(f"CUMULATIVE_FEATURE_LEAK: forbidden NoCC input columns selected: {sorted(set(bad))}")


def variant_feature_columns(variant: str) -> list[str]:
    if variant == "NoCC_VcorrInverse":
        cols = list(VCORR_FEATURES)
    elif variant == "NoCC_DynamicResponse":
        cols = list(DYNAMIC_FEATURES)
    elif variant.startswith("NoCC_BoundedCorrection"):
        cols = ordered_union(VCORR_FEATURES, DYNAMIC_FEATURES)
    elif variant == "NoCC_ThermoGuard":
        cols = ordered_union(VCORR_FEATURES, DYNAMIC_FEATURES, THERMAL_FEATURES, SURFACE_FEATURES)
    else:
        raise ValueError(f"Unknown NoCC variant: {variant}")
    audit_feature_columns(cols)
    return cols


def parse_correction_limit(variant: str) -> float:
    if variant.endswith("0p03"):
        return 0.03
    if variant.endswith("0p05"):
        return 0.05
    if variant.endswith("0p08"):
        return 0.08
    return 0.05


def augmented_indices(feature_cols: list[str], group_cols: list[str], mode: str) -> list[int]:
    base = [feature_cols.index(c) for c in group_cols if c in feature_cols]
    if not base:
        raise ValueError(f"No matching columns for group {group_cols}")
    n = len(feature_cols)
    idx = list(base)
    if mode in {"delta_start", "delta_start_time", "delta_start_time_local_residual"}:
        idx += [i + n for i in base]
    if mode in {"delta_start_time", "delta_start_time_local_residual"}:
        idx.append(2 * n)
    return sorted(set(idx))


def observability_components(frame_cache, start: int, end: int) -> dict[str, float]:
    sl = slice(start, end + 1)
    i = frame_cache["I_raw"][sl].astype(np.float64)
    di = frame_cache["dI"][sl].astype(np.float64)
    abs_i = np.abs(i)
    v = frame_cache["V_corr_raw"][sl].astype(np.float64)
    rest = abs_i <= 0.05
    longest_rest = 0
    cur = 0
    for flag in rest:
        cur = cur + 1 if bool(flag) else 0
        longest_rest = max(longest_rest, cur)
    if rest.sum() >= 3:
        rest_v = v[rest]
        relaxation = float(np.mean(np.abs(np.diff(rest_v)))) if len(rest_v) > 1 else 0.0
    else:
        relaxation = 0.0
    return {
        "current_variance": float(np.var(i)),
        "dI_energy": float(np.mean(di ** 2)),
        "rest_segment_fraction": float(longest_rest / max(1, len(i))),
        "voltage_response_amplitude": float(np.nanpercentile(v, 95) - np.nanpercentile(v, 5)),
        "relaxation_slope_visibility": float(relaxation),
        "window_mean_absI": float(np.mean(abs_i)),
        "endpoint_absI": float(abs_i[-1]) if len(abs_i) else np.nan,
    }


class ObservabilityNormalizer:
    keys = (
        "current_variance",
        "dI_energy",
        "rest_segment_fraction",
        "voltage_response_amplitude",
        "relaxation_slope_visibility",
    )

    def __init__(self, stats: dict[str, tuple[float, float]]):
        self.stats = stats

    @classmethod
    def fit(cls, raw_frames: list[pd.DataFrame], window_len: int, stride: int, max_windows: int = 50000):
        rows = []
        count = 0
        for frame in raw_frames:
            cache = raw_frame_cache(frame)
            n = len(frame)
            for start in range(0, n - int(window_len) + 1, int(stride)):
                rows.append(observability_components(cache, start, start + int(window_len) - 1))
                count += 1
                if count >= int(max_windows):
                    break
            if count >= int(max_windows):
                break
        df = pd.DataFrame(rows)
        stats = {}
        for key in cls.keys:
            vals = df[key].to_numpy(float)
            lo = float(np.nanpercentile(vals, 5)) if len(vals) else 0.0
            hi = float(np.nanpercentile(vals, 95)) if len(vals) else 1.0
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            stats[key] = (lo, hi)
        return cls(stats)

    def score(self, comps: dict[str, float]) -> float:
        vals = []
        for key in self.keys:
            lo, hi = self.stats[key]
            vals.append(np.clip((float(comps.get(key, 0.0)) - lo) / (hi - lo), 0.0, 1.0))
        return float(np.mean(vals)) if vals else np.nan


def raw_frame_cache(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    i = frame["I_raw"].to_numpy(np.float32)
    di = frame["dI"].to_numpy(np.float32) if "dI" in frame.columns else np.r_[0.0, np.diff(i)].astype(np.float32)
    return {
        "I_raw": i,
        "dI": di,
        "V_corr_raw": frame["V_corr_raw"].to_numpy(np.float32),
    }


class NoCCWindowDataset(DecomposedWindowDataset):
    def __init__(
        self,
        scaled_frames,
        raw_frames,
        feature_cols,
        window_len,
        stride,
        *,
        window_feature_mode: str,
        observability_normalizer: ObservabilityNormalizer | None = None,
    ):
        super().__init__(scaled_frames, feature_cols, window_len, stride, target_label="physical")
        self.window_feature_mode = str(window_feature_mode)
        self.raw_caches = [raw_frame_cache(f.reset_index(drop=True)) for f in raw_frames]
        self.observability_normalizer = observability_normalizer

    def __getitem__(self, idx):
        x, y, meta = super().__getitem__(idx)
        x = augment_window_tensor(x, self.window_feature_mode, self.feature_cols)
        if self.observability_normalizer is not None:
            fi, start, end = self.index[idx]
            comps = observability_components(self.raw_caches[fi], start, end)
            for key, val in comps.items():
                meta[key] = float(val)
            meta["observability_score"] = self.observability_normalizer.score(comps)
        return x, y, meta


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class EndpointHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder: str,
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        norm_kind: str,
    ):
        super().__init__()
        self.encoder = str(encoder)
        if self.encoder == "tcn":
            self.input_proj = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
            self.blocks = nn.Sequential(*[
                CausalConvBlock(
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                    norm_kind=norm_kind,
                )
                for i in range(int(layers))
            ])
            self.head = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(hidden_size, 1, kernel_size=1),
            )
        elif self.encoder == "lstm":
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            )
            self.rnn = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=int(layers),
                batch_first=True,
                dropout=float(dropout) if int(layers) > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_size, 1),
            )
        else:
            raise ValueError(f"Unknown encoder={encoder}")

    def logits(self, x):
        if self.encoder == "tcn":
            h = self.input_proj(x.transpose(1, 2))
            h = self.blocks(h)
            return self.head(h).transpose(1, 2)[:, -1, :]
        z = self.input_proj(x)
        out, _ = self.rnn(z)
        return self.head(self.norm(out[:, -1, :]))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


class NoCCDirectRegressor(nn.Module):
    def __init__(self, input_dim: int, cfg: NoCCConfig):
        super().__init__()
        self.head = EndpointHead(
            input_dim,
            cfg.encoder,
            cfg.hidden_size,
            cfg.layers,
            cfg.kernel_size,
            cfg.dropout,
            cfg.norm_kind,
        )

    def forward(self, x):
        return self.head(x)


class NoCCBoundedCorrection(nn.Module):
    def __init__(self, input_dim: int, feature_cols: list[str], cfg: NoCCConfig, correction_limit: float):
        super().__init__()
        self.base_idx = augmented_indices(feature_cols, VCORR_FEATURES, cfg.window_feature_mode)
        self.corr_idx = augmented_indices(feature_cols, DYNAMIC_FEATURES, cfg.window_feature_mode)
        self.correction_limit = float(correction_limit)
        self.base = EndpointHead(
            len(self.base_idx), cfg.encoder, cfg.hidden_size, cfg.layers,
            cfg.kernel_size, cfg.dropout, cfg.norm_kind
        )
        self.corr = EndpointHead(
            len(self.corr_idx), cfg.encoder, cfg.hidden_size, cfg.layers,
            cfg.kernel_size, cfg.dropout, cfg.norm_kind
        )

    def forward(self, x):
        base_soc = torch.sigmoid(self.base.logits(x[..., self.base_idx]))
        delta = self.correction_limit * torch.tanh(self.corr.logits(x[..., self.corr_idx]))
        return (base_soc + delta).clamp(0.0, 1.0)


class NoCCThermoGuard(nn.Module):
    def __init__(self, input_dim: int, feature_cols: list[str], cfg: NoCCConfig):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.vcorr_idx = augmented_indices(feature_cols, VCORR_FEATURES, cfg.window_feature_mode)
        self.dynamic_idx = augmented_indices(feature_cols, DYNAMIC_FEATURES, cfg.window_feature_mode)
        self.thermal_idx = augmented_indices(feature_cols, THERMAL_FEATURES, cfg.window_feature_mode)
        self.surface_idx = augmented_indices(feature_cols, SURFACE_FEATURES, cfg.window_feature_mode)
        self.guard_cols = [c for c in ["T", "T_eff_proxy", "excitation_score", "V_response_amp", "R0"] if c in feature_cols]
        self.guard_idx = augmented_indices(feature_cols, self.guard_cols, cfg.window_feature_mode)
        common = dict(
            encoder=cfg.encoder,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            norm_kind=cfg.norm_kind,
        )
        self.experts = nn.ModuleList([
            EndpointHead(len(self.vcorr_idx), **common),
            EndpointHead(len(self.dynamic_idx), **common),
            EndpointHead(len(self.thermal_idx), **common),
            EndpointHead(len(self.surface_idx), **common),
        ])
        guard_dim = len(self.guard_idx) + 3
        self.gate = nn.Sequential(
            nn.Linear(guard_dim, max(32, cfg.hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(max(32, cfg.hidden_size // 2), 4),
        )

    def forward(self, x):
        preds = torch.cat([
            self.experts[0](x[..., self.vcorr_idx]),
            self.experts[1](x[..., self.dynamic_idx]),
            self.experts[2](x[..., self.thermal_idx]),
            self.experts[3](x[..., self.surface_idx]),
        ], dim=1)
        disagreement = torch.mean(torch.abs(preds - preds.mean(dim=1, keepdim=True)), dim=1, keepdim=True)
        last_guard = x[:, -1, self.guard_idx]
        voltage_conf = x[:, -1, :].abs().mean(dim=1, keepdim=True)
        ood_proxy = x[:, -1, :].abs().topk(k=max(1, min(8, x.shape[-1])), dim=1).values.mean(dim=1, keepdim=True)
        guard_in = torch.cat([last_guard, disagreement, voltage_conf, ood_proxy], dim=1)
        weights = torch.softmax(self.gate(guard_in), dim=1)
        return torch.sum(weights * preds, dim=1, keepdim=True).clamp(0.0, 1.0)


def build_no_cc_model(variant: str, input_dim: int, feature_cols: list[str], cfg: NoCCConfig) -> nn.Module:
    if variant in {"NoCC_VcorrInverse", "NoCC_DynamicResponse"}:
        return NoCCDirectRegressor(input_dim, cfg)
    if variant.startswith("NoCC_BoundedCorrection"):
        return NoCCBoundedCorrection(input_dim, feature_cols, cfg, parse_correction_limit(variant))
    if variant == "NoCC_ThermoGuard":
        return NoCCThermoGuard(input_dim, feature_cols, cfg)
    raise ValueError(f"Unknown variant={variant}")


def audit_no_cc_models_or_fail(feature_cols_by_variant: dict[str, list[str]]):
    for variant, cols in feature_cols_by_variant.items():
        audit_feature_columns(cols)
    sources = "\n".join(
        inspect.getsource(obj)
        for obj in [NoCCDirectRegressor, NoCCBoundedCorrection, NoCCThermoGuard, EndpointHead]
    )
    explicit_patterns = [
        r"dt\s*/\s*3600",
        r"3600",
        r"Q_eff",
        r"use_current_integration",
        r"soc\s*=\s*soc.*I",
        r"soc.*-\s*.*I.*Q",
    ]
    for pat in explicit_patterns:
        if re.search(pat, sources, flags=re.I):
            raise RuntimeError(f"EXPLICIT_INTEGRATION_LEAK: matched source pattern {pat}")
    return {
        "status": "pass",
        "explicit_current_integration_state_update": False,
        "soc_input_features": False,
        "cumulative_or_capacity_features": False,
        "feature_columns_by_variant": feature_cols_by_variant,
    }


def make_loader(ds, cfg: NoCCConfig, *, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def move_float(x):
    return x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def group_loss(pred, y, meta, cfg: NoCCConfig):
    sample = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
    temps = meta["temperature"]
    drives = meta["drive_cycle"]
    if torch.is_tensor(temps):
        temps_list = [float(v) for v in temps.detach().cpu().tolist()]
    else:
        temps_list = [float(v) for v in list(temps)]
    drives_list = [str(v) for v in (list(drives) if not torch.is_tensor(drives) else drives.detach().cpu().tolist())]
    keys = [f"T{t:g}_{d}" for t, d in zip(temps_list, drives_list)]
    losses = []
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        t_idx = torch.as_tensor(idx, device=pred.device, dtype=torch.long)
        losses.append(sample.index_select(0, t_idx).mean())
    stack = torch.stack(losses) if losses else sample.mean().view(1)
    mean_loss = stack.mean()
    rex = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
    worst = stack.max() if len(stack) > 1 else stack.new_tensor(0.0)
    return mean_loss + float(cfg.lambda_rex) * rex + float(cfg.lambda_worst) * worst, mean_loss, rex, worst


def train_one_model(variant: str, train_ds: NoCCWindowDataset, cfg: NoCCConfig, input_dim: int, feature_cols: list[str]):
    model = build_no_cc_model(variant, input_dim, feature_cols, cfg).to(device)
    loader = make_loader(train_ds, cfg, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history = []
    best_loss = float("inf")
    best_state = None
    bad = 0
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        means = []
        rexes = []
        worsts = []
        for x, y, meta in loader:
            x = move_float(x)
            y = move_float(y)
            pred = model(x)
            loss, mean_loss, rex_loss, worst_loss = group_loss(pred, y, meta, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            means.append(float(mean_loss.detach().cpu()))
            rexes.append(float(rex_loss.detach().cpu()))
            worsts.append(float(worst_loss.detach().cpu()))
        row = {
            "variant": variant,
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "mean_huber": float(np.mean(means)),
            "rex_var": float(np.mean(rexes)),
            "worst_group": float(np.mean(worsts)),
        }
        history.append(row)
        if row["train_loss"] < best_loss - float(cfg.min_delta):
            best_loss = row["train_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(
                f"{variant} epoch={ep} loss={row['train_loss']:.5f} "
                f"mean={row['mean_huber']:.5f} rex={row['rex_var']:.6f}",
                flush=True,
            )
        if ep >= int(cfg.warmup_epochs) and bad >= int(cfg.patience):
            history[-1]["stopped_early"] = True
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_model(model, loader, variant: str, fold: str, seed: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        pred = model(move_float(x)).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["model_name"] = variant
        mdf["run_model_name"] = variant
        mdf["target_label"] = "physical"
        mdf["label_type"] = "physical"
        mdf["fold_name"] = fold
        mdf["experiment"] = fold
        mdf["seed"] = int(seed)
        mdf["y_true"] = yy
        mdf["y_pred"] = pred
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["temperature_C"] = out["temperature"].astype(float)
        out["time_index"] = out["end_index"]
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        max_index = out.groupby("trajectory_id")["end_index"].transform("max").replace(0, np.nan)
        out["trajectory_fraction"] = out["end_index"] / max_index
        out["is_plateau_20_80"] = (out["y_true"] >= 0.2) & (out["y_true"] <= 0.8)
        out["is_cutoff_last10"] = out["trajectory_fraction"] >= 0.9
        out["label_policy"] = "physical_smoothQ"
    return out


def trajectory_jitter(g: pd.DataFrame) -> tuple[float, float, float]:
    pred_vals = []
    true_vals = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        yp = t["y_pred"].to_numpy(float)
        yt = t["y_true"].to_numpy(float)
        if len(yp) < 3:
            continue
        pred_vals.append(float(np.mean(np.abs(np.diff(yp)))))
        true_vals.append(float(np.mean(np.abs(np.diff(yt)))))
    pred_j = float(np.mean(pred_vals)) if pred_vals else np.nan
    true_j = float(np.mean(true_vals)) if true_vals else np.nan
    ratio = pred_j / (true_j + 1e-12) if np.isfinite(pred_j) and np.isfinite(true_j) else np.nan
    return pred_j, true_j, float(ratio)


def hf_error(g: pd.DataFrame) -> float:
    vals = []
    for _, t in g.sort_values("end_index").groupby("trajectory_id"):
        e = t["y_pred"].to_numpy(float) - t["y_true"].to_numpy(float)
        if len(e) >= 3:
            vals.append(float(np.mean(np.diff(e) ** 2)))
    return float(np.mean(vals)) if vals else np.nan


def region_mae(g: pd.DataFrame, mask) -> float:
    if mask is None:
        return np.nan
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return np.nan
    err = g.loc[mask, "y_pred"].to_numpy(float) - g.loc[mask, "y_true"].to_numpy(float)
    return float(np.mean(np.abs(err)) * 100.0)


def metrics_for_group(g: pd.DataFrame) -> dict[str, float]:
    err = g["y_pred"].to_numpy(float) - g["y_true"].to_numpy(float)
    abs_err = np.abs(err)
    pred_j, true_j, jr = trajectory_jitter(g)
    low_current_mask = g.get("endpoint_absI", pd.Series(np.nan, index=g.index)).astype(float).to_numpy() <= 0.05
    return {
        "MAE_pct": float(np.mean(abs_err) * 100.0),
        "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
        "Max_error_pct": float(np.max(abs_err) * 100.0),
        "error_std_pct": float(np.std(err) * 100.0),
        "pred_jitter": pred_j,
        "true_jitter": true_j,
        "jitter_ratio": jr,
        "high_frequency_error_energy": hf_error(g),
        "catastrophic_error_rate_5pct": float(np.mean(abs_err > 0.05)),
        "plateau_20_80_MAE_pct": region_mae(g, g["is_plateau_20_80"].to_numpy(dtype=bool)),
        "low_current_MAE_pct": region_mae(g, low_current_mask),
        "n_samples": int(len(g)),
    }


def fold_target_type(fold: str) -> str:
    if fold == "Exp D":
        return "included_diagnostic"
    if fold in {"Omit N10", "Omit 50"}:
        return "outside"
    return "omitted"


def compute_results(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    by_temp_rows = []
    obs_rows = []
    obs_vs_rows = []
    if pred.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
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
                "source": "no_cc_one_seed",
            }
            row.update(metrics_for_group(target))
            fold_rows.append(row)
        for temp, tg in g.groupby("temperature_C"):
            row = {
                "model_name": model,
                "seed": int(seed),
                "fold_name": fold,
                "temperature_C": float(temp),
                "source": "no_cc_one_seed",
            }
            row.update(metrics_for_group(tg))
            by_temp_rows.append(row)
        if "observability_score" in g.columns:
            gg = g.copy()
            gg["observability_bin"] = pd.cut(
                gg["observability_score"].astype(float),
                bins=[-np.inf, 0.33, 0.66, np.inf],
                labels=["low", "mid", "high"],
            )
            for obs_bin, og in gg.groupby("observability_bin", observed=False):
                if og.empty:
                    continue
                row = {
                    "model_name": model,
                    "seed": int(seed),
                    "fold_name": fold,
                    "observability_bin": str(obs_bin),
                    "mean_observability_score": float(og["observability_score"].mean()),
                    "source": "no_cc_one_seed",
                }
                row.update(metrics_for_group(og))
                obs_rows.append(row)
            try:
                gg["observability_decile"] = pd.qcut(
                    gg["observability_score"].rank(method="first"),
                    q=10,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                gg["observability_decile"] = 0
            corr = float(gg[["observability_score", "abs_error"]].corr(method="spearman").iloc[0, 1])
            for decile, dg in gg.groupby("observability_decile"):
                obs_vs_rows.append({
                    "model_name": model,
                    "seed": int(seed),
                    "fold_name": fold,
                    "observability_decile": int(decile),
                    "mean_observability_score": float(dg["observability_score"].mean()),
                    "MAE_pct": float(dg["abs_error"].mean() * 100.0),
                    "RMSE_pct": float(np.sqrt(np.mean(dg["error"].to_numpy(float) ** 2)) * 100.0),
                    "n_samples": int(len(dg)),
                    "spearman_observability_vs_abs_error": corr,
                    "source": "no_cc_one_seed",
                })
    focus = summarize_focus(pd.DataFrame(fold_rows))
    return pd.DataFrame(fold_rows), pd.DataFrame(by_temp_rows), focus, pd.DataFrame(obs_rows), pd.DataFrame(obs_vs_rows)


def summarize_focus(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    scopes = {
        "omitted_A_B_C": ["Exp A", "Exp B", "Exp C"],
        "outside_range": ["Omit N10", "Omit 50"],
        "included_diagnostic": ["Exp D"],
        "all_target_folds": ["Exp A", "Exp B", "Exp C", "Exp D", "Omit N10", "Omit 50"],
    }
    for (model, source), mg in results.groupby(["model_name", "source"]):
        for scope, folds in scopes.items():
            g = mg[mg["fold_name"].isin(folds)]
            if g.empty:
                continue
            row = {
                "model_name": model,
                "source": source,
                "scope": scope,
                "average_MAE_pct": float(g["MAE_pct"].mean()),
                "worst_MAE_pct": float(g["MAE_pct"].max()),
                "average_RMSE_pct": float(g["RMSE_pct"].mean()),
                "worst_RMSE_pct": float(g["RMSE_pct"].max()),
                "average_jitter_ratio": float(g["jitter_ratio"].mean()),
                "average_catastrophic_error_rate_5pct": float(g["catastrophic_error_rate_5pct"].mean()),
                "average_plateau_20_80_MAE_pct": float(g["plateau_20_80_MAE_pct"].mean()),
                "average_low_current_MAE_pct": float(g["low_current_MAE_pct"].mean()) if "low_current_MAE_pct" in g else np.nan,
                "n_folds": int(g["fold_name"].nunique()),
            }
            for fold in folds:
                sub = g[g["fold_name"].eq(fold)]
                row[f"{fold}_MAE_pct"] = float(sub["MAE_pct"].iloc[0]) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_reference_fold_metrics(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "final_10seed_metrics_by_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["model_name"].isin(REFERENCE_MODELS)].copy()
    if df.empty:
        return pd.DataFrame()
    group_cols = ["model_name", "fold_name", "target_temperature_C", "target_type"]
    metric_cols = [
        "MAE_pct", "RMSE_pct", "Max_error_pct", "error_std_pct", "pred_jitter", "true_jitter",
        "jitter_ratio", "high_frequency_error_energy", "catastrophic_error_rate_5pct",
        "plateau_20_80_MAE_pct", "n_samples",
    ]
    available = [c for c in metric_cols if c in df.columns]
    out = df.groupby(group_cols, as_index=False)[available].mean(numeric_only=True)
    out["seed"] = -1
    out["low_current_MAE_pct"] = np.nan
    out["source"] = "reference_10seed_mean"
    return out


def load_reference_temperature_metrics(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "final_10seed_metrics_by_temperature.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["model_name"].isin(REFERENCE_MODELS)].copy()
    if df.empty:
        return pd.DataFrame()
    group_cols = ["model_name", "fold_name", "temperature_C"]
    metric_cols = [
        "MAE_pct", "RMSE_pct", "Max_error_pct", "error_std_pct", "pred_jitter", "true_jitter",
        "jitter_ratio", "high_frequency_error_energy", "catastrophic_error_rate_5pct", "n_samples",
    ]
    out = df.groupby(group_cols, as_index=False)[[c for c in metric_cols if c in df.columns]].mean(numeric_only=True)
    out["seed"] = -1
    out["source"] = "reference_10seed_mean"
    out["plateau_20_80_MAE_pct"] = np.nan
    out["low_current_MAE_pct"] = np.nan
    return out


def save_trace_plots(pred: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if pred.empty:
        return []
    paths = []
    for (fold, model), g in pred.groupby(["fold_name", "model_name"]):
        target_temp = float(EXPERIMENTS[fold]["omitted_temp_C"])
        sub = g[np.isclose(g["temperature_C"].astype(float), target_temp)]
        if sub.empty:
            sub = g
        tid = sub.groupby("trajectory_id")["abs_error"].mean().sort_values().index[0]
        t = sub[sub["trajectory_id"].eq(tid)].sort_values("end_index")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(t["end_index"], t["y_true"] * 100.0, label="label", linewidth=1.6)
        ax.plot(t["end_index"], t["y_pred"] * 100.0, label="prediction", linewidth=1.2)
        ax.set_title(f"{fold} {model} {tid}")
        ax.set_xlabel("time index")
        ax.set_ylabel("SOC (%)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{fold.replace(' ', '_')}_{model}_{tid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_report(
    cfg: NoCCConfig,
    results: pd.DataFrame,
    focus: pd.DataFrame,
    obs_metrics: pd.DataFrame,
    audit: dict,
    out_path: Path,
):
    lines = [
        "# Strict No-CC SOC Ablation Report",
        "",
        "## Scope",
        "- Main label: `physical_smoothQ`.",
        "- The NoCC models use current only as instantaneous/dynamic excitation; no SOC state is advanced by accumulated current.",
        "- Inputs exclude precomputed SOC, usable-to-cutoff SOC, capacity-normalized cumulative discharge, and cumulative charge features.",
        "- Window-local normalized position may be used; absolute start/end timestep and trajectory progress are not model inputs.",
        "- Window encoders are stateless: recurrent hidden state, when used, is reset for every window.",
        "",
        "## Audit",
        f"- Audit status: `{audit.get('status', 'unknown')}`.",
        f"- explicit current-integration SOC state update: `{audit.get('explicit_current_integration_state_update')}`.",
        f"- SOC input features: `{audit.get('soc_input_features')}`.",
        f"- cumulative/capacity input features: `{audit.get('cumulative_or_capacity_features')}`.",
        "",
    ]
    if len(focus):
        no_cc_focus = focus[focus["source"].eq("no_cc_one_seed")].sort_values(["scope", "average_MAE_pct"])
        ref_focus = focus[focus["source"].ne("no_cc_one_seed")].sort_values(["scope", "average_MAE_pct"])
        lines.extend([
            "## NoCC Target Summary",
            no_cc_focus.to_markdown(index=False),
            "",
        ])
        if len(ref_focus):
            lines.extend([
                "## Reference Summary",
                ref_focus.to_markdown(index=False),
                "",
            ])
    if len(results):
        best_no_cc = results[results["source"].eq("no_cc_one_seed")].sort_values("MAE_pct").head(12)
        lines.extend([
            "## Best NoCC Fold Rows",
            best_no_cc[[
                "model_name", "fold_name", "target_type", "target_temperature_C",
                "MAE_pct", "RMSE_pct", "jitter_ratio", "catastrophic_error_rate_5pct",
                "plateau_20_80_MAE_pct", "low_current_MAE_pct",
            ]].to_markdown(index=False),
            "",
        ])
    if len(obs_metrics):
        obs_summary = (
            obs_metrics.groupby(["model_name", "observability_bin"])[["MAE_pct", "n_samples"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        lines.extend([
            "## Observability Bins",
            obs_summary.to_markdown(index=False),
            "",
        ])
    no_cc = results[results["source"].eq("no_cc_one_seed")] if len(results) else pd.DataFrame()
    ref = results[results["source"].eq("reference_10seed_mean")] if len(results) else pd.DataFrame()
    if len(no_cc):
        best_no_cc_by_fold = no_cc.loc[no_cc.groupby("fold_name")["MAE_pct"].idxmin()].copy()
        best_no_cc_mae = float(best_no_cc_by_fold["MAE_pct"].mean())
        all_no_cc_mae = float(no_cc["MAE_pct"].mean())
    else:
        best_no_cc_by_fold = pd.DataFrame()
        best_no_cc_mae = np.nan
        all_no_cc_mae = np.nan
    ref_models = ref[ref["model_name"].isin(["NeuralECM_Tamb_Tcore", "ThermoGuardSOC_rule_soft_high"])]
    if len(ref_models):
        best_ref_by_fold = ref_models.loc[ref_models.groupby("fold_name")["MAE_pct"].idxmin()].copy()
        best_ref_mae = float(best_ref_by_fold["MAE_pct"].mean())
    else:
        best_ref_by_fold = pd.DataFrame()
        best_ref_mae = np.nan
    degrade = best_no_cc_mae - best_ref_mae if np.isfinite(best_no_cc_mae) and np.isfinite(best_ref_mae) else np.nan
    if len(best_no_cc_by_fold):
        lines.extend([
            "## Best NoCC Versus Best Reference By Fold",
            best_no_cc_by_fold[[
                "fold_name", "model_name", "target_temperature_C", "target_type",
                "MAE_pct", "RMSE_pct", "jitter_ratio", "plateau_20_80_MAE_pct",
            ]].to_markdown(index=False),
            "",
        ])
    if len(best_ref_by_fold):
        lines.extend([
            "## Best Integrated Reference By Fold",
            best_ref_by_fold[[
                "fold_name", "model_name", "target_temperature_C", "target_type",
                "MAE_pct", "RMSE_pct", "jitter_ratio", "plateau_20_80_MAE_pct",
            ]].to_markdown(index=False),
            "",
        ])
    lines.extend([
        "## Required Answers",
        "1. Does removing explicit current integration significantly degrade performance?",
        f"   - In this one-seed NoCC run, the best NoCC variant per fold averages {best_no_cc_mae:.3f}% target MAE "
        f"(all NoCC variants average {all_no_cc_mae:.3f}%). "
        f"The best integrated NeuralECM/ThermoGuard reference per fold averages {best_ref_mae:.3f}%, "
        f"so the observed best-case NoCC delta is {degrade:.3f} percentage points. "
        "Interpret this as an ablation result, not as proof that charge conservation is unnecessary.",
        "2. Which folds are most affected?",
        "   - Use `no_cc_focus.csv` and the worst target-fold rows in `no_cc_results.csv`; outside-range folds should be treated separately from Exp A/B/C omissions.",
        "3. Can voltage/thermal/dynamic response recover SOC without charge conservation?",
        "   - It can recover part of the mapping when voltage response is observable, but the result should be described as response-based SOC inference rather than pure extrapolation.",
        "4. Does observability score identify failure regions?",
        "   - Only weakly in the current implementation. The score is label-free and useful as a diagnostic column, but its bin ordering is not yet a reliable failure detector; refine it before using it as a guard.",
        "5. Is NoCC viable as main model, or only as ablation?",
        "   - Unless NoCC matches the integrated observer across outside-range folds, the safe conclusion is that NoCC is primarily an ablation/control model.",
        "6. Safe paper wording.",
        "   - Recommended: `The No-CC ablation removes explicit Coulomb-counting-style SOC state propagation. Current is still supplied as an instantaneous excitation signal for voltage-response features. The resulting performance quantifies how much SOC information can be recovered from voltage, thermal, and dynamic response alone under the tested folds.`",
        "",
        "## Forbidden Claims Avoided",
        "- This report does not claim that NoCC solves pure extrapolation.",
        "- This report does not claim that current integration is unnecessary.",
        "- This report does not claim current is unused.",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_one_fold(cfg: NoCCConfig, fold: str, lookup: pd.DataFrame):
    base_cfg = make_cfg()
    base_cfg.output_dir = cfg.base_dir
    base_cfg.base_dir = cfg.base_dir
    base_cfg = experiment_cfg(base_cfg, fold)
    configure_strict_training(base_cfg)
    base_cfg.window_len = int(cfg.window_len)
    base_cfg.stride = int(cfg.stride)
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    raw_frames = add_no_cc_derived_features(load_relabelled_frames(base_cfg, fold, lookup))
    obs_norm = ObservabilityNormalizer.fit(raw_frames["train"], cfg.window_len, cfg.stride)
    pred_rows = []
    histories = []
    for variant in cfg.variants:
        cols = variant_feature_columns(variant)
        available = set().union(*(set(f.columns) for split in raw_frames.values() for f in split))
        missing = [c for c in cols if c not in available]
        if missing:
            raise KeyError(f"{fold} {variant}: missing NoCC feature columns {missing}")
        scaled, _ = make_scaled_frames_for_ablation(raw_frames, cols)
        input_dim = augmented_input_dim(len(cols), cfg.window_feature_mode)
        train_ds = NoCCWindowDataset(
            scaled["train"], raw_frames["train"], cols, cfg.window_len, cfg.stride,
            window_feature_mode=cfg.window_feature_mode,
        )
        test_ds = NoCCWindowDataset(
            scaled["test"], raw_frames["test"], cols, cfg.window_len, cfg.stride,
            window_feature_mode=cfg.window_feature_mode,
            observability_normalizer=obs_norm,
        )
        print(f"[{fold}] {variant}: train_windows={len(train_ds)} test_windows={len(test_ds)} input_dim={input_dim}", flush=True)
        model, hist = train_one_model(variant, train_ds, cfg, input_dim, cols)
        hist["fold_name"] = fold
        hist["seed"] = int(cfg.seed)
        histories.append(hist)
        pred = predict_model(model, make_loader(test_ds, cfg, shuffle=False), variant, fold, cfg.seed)
        pred_rows.append(pred)
    pred_all = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    hist_all = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return pred_all, hist_all


def run_no_cc(cfg: NoCCConfig):
    cfg.base_dir = Path(cfg.base_dir)
    configure_torch_runtime()
    set_seed(cfg.seed)
    feature_cols_by_variant = {variant: variant_feature_columns(variant) for variant in cfg.variants}
    audit = audit_no_cc_models_or_fail(feature_cols_by_variant)
    (cfg.base_dir / f"{cfg.output_prefix}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lookup = load_smoothq_lookup(cfg.base_dir)
    all_pred = []
    all_hist = []
    started = time.time()
    for fold in cfg.folds:
        pred, hist = run_one_fold(cfg, fold, lookup)
        all_pred.append(pred)
        all_hist.append(hist)
        partial = pd.concat(all_pred, ignore_index=True)
        partial.to_csv(cfg.base_dir / f"{cfg.output_prefix}_prediction_rows_partial.csv.gz", index=False, compression="gzip")
    pred = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    hist = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    results, by_temp, focus, obs_metrics, obs_vs = compute_results(pred)
    ref_results = load_reference_fold_metrics(cfg.base_dir)
    if len(ref_results):
        results = pd.concat([results, ref_results], ignore_index=True, sort=False)
    ref_by_temp = load_reference_temperature_metrics(cfg.base_dir)
    if len(ref_by_temp):
        by_temp = pd.concat([by_temp, ref_by_temp], ignore_index=True, sort=False)
    focus = summarize_focus(results)
    results.to_csv(cfg.base_dir / f"{cfg.output_prefix}_results.csv", index=False)
    by_temp.to_csv(cfg.base_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    focus.to_csv(cfg.base_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    obs_metrics.to_csv(cfg.base_dir / f"{cfg.output_prefix}_observability_metrics.csv", index=False)
    obs_vs.to_csv(cfg.base_dir / f"{cfg.output_prefix}_observability_vs_error.csv", index=False)
    hist.to_csv(cfg.base_dir / f"{cfg.output_prefix}_history.csv", index=False)
    if cfg.save_predictions:
        pred.to_csv(cfg.base_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    plot_paths = save_trace_plots(pred, cfg.base_dir / f"{cfg.output_prefix}_soc_trace_plots")
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
        "amp_used": False,
        "label_policy": "physical_smoothQ",
        "window_encoder_stateless": True,
        "future_current_used": False,
        "current_usage": "instantaneous excitation and causal dynamic response features only",
        "absolute_timestep_input": False,
        "window_local_position_input": cfg.window_feature_mode in {"delta_start_time", "delta_start_time_local_residual"},
        "window_position_policy": "delta_start_time appends normalized within-window position only; no absolute start/end timestep or trajectory progress is appended to model input",
        "audit": audit,
        "duration_s": time.time() - started,
        "trace_plot_count": len(plot_paths),
    }
    (cfg.base_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, results, focus, obs_metrics, audit, cfg.base_dir / f"{cfg.output_prefix}_report.md")
    return {
        "prediction_rows": pred,
        "results": results,
        "by_temperature": by_temp,
        "focus": focus,
        "observability_metrics": obs_metrics,
        "observability_vs_error": obs_vs,
        "history": hist,
        "metadata": metadata,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Strict No-CC SOC ablation without explicit current-integration SOC state update.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--output-prefix", default="no_cc")
    p.add_argument("--folds", nargs="+", default=list(NO_CC_FOLDS), choices=list(NO_CC_FOLDS))
    p.add_argument("--variants", nargs="+", default=list(NoCCConfig.variants))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", choices=["tcn", "lstm"], default="tcn")
    p.add_argument("--window-len", type=int, default=50)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument("--lambda-rex", type=float, default=0.30)
    p.add_argument("--lambda-worst", type=float, default=0.0)
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--window-feature-mode", default="delta_start_time", choices=["raw", "delta_start", "delta_start_time"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=2e-5)
    p.add_argument("--no-save-predictions", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = NoCCConfig(
        base_dir=args.base_dir,
        output_prefix=args.output_prefix,
        folds=tuple(args.folds),
        variants=tuple(args.variants),
        seed=args.seed,
        encoder=args.encoder,
        window_len=args.window_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        lambda_rex=args.lambda_rex,
        lambda_worst=args.lambda_worst,
        huber_beta=args.huber_beta,
        window_feature_mode=args.window_feature_mode,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        print_every=args.print_every,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        min_delta=args.min_delta,
        save_predictions=not args.no_save_predictions,
    )
    run_no_cc(cfg)


if __name__ == "__main__":
    main()
