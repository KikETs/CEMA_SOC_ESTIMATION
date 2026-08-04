from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import make_cfg
from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .training import (
    attach_prediction_features,
    build_prediction_feature_lookup,
    make_scaled_frames_for_ablation,
)
from .models import DecomposedWindowDataset, build_lstm_soc_model, collate_meta_to_frame
from .extrapolation_robustness import temperature_balanced_loader
from .variance_control import R5_GATED_FEATURES, _overall_metrics, variance_by_temperature


FORBIDDEN_INPUT_TOKENS = ("SOC", "soc", "cumulative", "Q_ref", "q_cutoff")
BASE_DEEP_FEATURES = list(R5_GATED_FEATURES)
EXTENDED_DEEP_FEATURES = BASE_DEEP_FEATURES + [
    "R0_x_V_pol",
    "T_x_V_pol",
    "R0_x_absI",
    "V_pol_x_abs_dI",
]
BAND_DEEP_FEATURES = EXTENDED_DEEP_FEATURES + [
    "V_residual_raw",
    "V_residual_low",
    "V_residual_mid",
    "V_residual_high",
    "V_residual_low_x_T",
    "V_residual_mid_x_absI",
    "V_residual_high_x_abs_dI",
]
BRANCH_DEEP_FEATURES = EXTENDED_DEEP_FEATURES + [
    "V_pol_fast_raw",
    "V_pol_mid_raw",
    "V_pol_slow_raw",
]
BRANCH_BAND_DEEP_FEATURES = BRANCH_DEEP_FEATURES + [
    c for c in BAND_DEEP_FEATURES if c not in EXTENDED_DEEP_FEATURES
]
SPECTRAL_BAND_DEEP_FEATURES = EXTENDED_DEEP_FEATURES + [
    "V_pol_fast_raw",
    "V_pol_mid_raw",
    "V_pol_slow_raw",
    "V_pol_lowfreq",
    "V_pol_midfreq",
    "V_pol_highfreq",
] + [
    c for c in BAND_DEEP_FEATURES if c not in EXTENDED_DEEP_FEATURES
]
FILTER_BASE_COLUMNS = ["V_raw", "V_corr_raw", "I_raw", "absI", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]
FILTER_TAUS = (10, 50, 200, 800)
FILTERED_DEEP_FEATURES = EXTENDED_DEEP_FEATURES + [
    f"{col}_ema{tau}" for col in FILTER_BASE_COLUMNS for tau in FILTER_TAUS
] + [
    f"{col}_dev_ema{tau}" for col in FILTER_BASE_COLUMNS for tau in (50, 200, 800)
]


@dataclass
class DeepNoLeakConfig:
    base_dir: Path = Path(".")
    output_prefix: str = "deep_no_leak"
    experiment: str = "Omit N10"
    feature_dir: str = ""
    seed: int = 0
    model_kind: str = "tcn"
    feature_set: str = "extended"
    window_feature_mode: str = "raw"
    window_len: int = 300
    stride: int = 2
    epochs: int = 80
    batch_size: int = 512
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 96
    layers: int = 6
    kernel_size: int = 5
    num_experts: int = 4
    norm_kind: str = "channel"
    dropout: float = 0.08
    lambda_rex: float = 0.4
    rex_group: str = "temperature"
    lambda_smooth: float = 0.0
    endpoint_loss_weight: float = 0.0
    lambda_worst: float = 0.0
    loss_kind: str = "mae"
    huber_beta: float = 0.02
    sequence_loss: bool = False
    trajectory_mode: bool = False
    num_workers: int = 0
    prefetch_factor: int = 4
    allow_tf32: bool = False
    print_every: int = 10


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def feature_columns(name: str) -> list[str]:
    if name == "base":
        cols = BASE_DEEP_FEATURES
    elif name == "extended":
        cols = EXTENDED_DEEP_FEATURES
    elif name == "bands":
        cols = BAND_DEEP_FEATURES
    elif name == "branch":
        cols = BRANCH_DEEP_FEATURES
    elif name == "branch_bands":
        cols = BRANCH_BAND_DEEP_FEATURES
    elif name == "spectral_bands":
        cols = SPECTRAL_BAND_DEEP_FEATURES
    elif name == "voltage_only":
        cols = ["V_raw", "V_corr_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]
    elif name == "filtered":
        cols = FILTERED_DEEP_FEATURES
    else:
        raise ValueError(f"Unknown feature_set={name}")
    bad = [c for c in cols if any(tok in c for tok in FORBIDDEN_INPUT_TOKENS)]
    if bad:
        raise AssertionError(f"Forbidden leak-prone input columns selected: {bad}")
    return list(cols)


def add_derived_features(frames: dict[str, list[pd.DataFrame]]) -> dict[str, list[pd.DataFrame]]:
    out = {}
    for split, split_frames in frames.items():
        out[split] = []
        for frame in split_frames:
            f = frame.copy()
            if "R0_x_V_pol" not in f.columns and {"R0", "V_pol_raw"}.issubset(f.columns):
                f["R0_x_V_pol"] = f["R0"].astype(float) * f["V_pol_raw"].astype(float)
            if "T_x_V_pol" not in f.columns and {"T", "V_pol_raw"}.issubset(f.columns):
                f["T_x_V_pol"] = f["T"].astype(float) * f["V_pol_raw"].astype(float)
            if "R0_x_absI" not in f.columns and {"R0", "absI"}.issubset(f.columns):
                f["R0_x_absI"] = f["R0"].astype(float) * f["absI"].astype(float)
            if "V_pol_x_abs_dI" not in f.columns and {"V_pol_raw", "dI"}.issubset(f.columns):
                f["V_pol_x_abs_dI"] = f["V_pol_raw"].astype(float) * f["dI"].abs().astype(float)
            if "V_residual_raw" not in f.columns and {"V_raw", "V_corr_raw"}.issubset(f.columns):
                residual = (
                    f["V_raw"].to_numpy(np.float64)
                    - f["V_corr_raw"].to_numpy(np.float64)
                )
                slow = _ema_causal(residual, cutoff_hz=0.003)
                mid_lp = _ema_causal(residual, cutoff_hz=0.03)
                f["V_residual_raw"] = residual.astype(np.float32)
                f["V_residual_low"] = slow.astype(np.float32)
                f["V_residual_mid"] = (mid_lp - slow).astype(np.float32)
                f["V_residual_high"] = (residual - mid_lp).astype(np.float32)
                if "T" in f.columns:
                    f["V_residual_low_x_T"] = f["V_residual_low"].astype(float) * f["T"].astype(float)
                if "absI" in f.columns:
                    f["V_residual_mid_x_absI"] = f["V_residual_mid"].astype(float) * f["absI"].astype(float)
                if "dI" in f.columns:
                    f["V_residual_high_x_abs_dI"] = f["V_residual_high"].astype(float) * f["dI"].abs().astype(float)
            for col in FILTER_BASE_COLUMNS:
                if col not in f.columns:
                    continue
                arr = f[col].to_numpy(np.float32)
                for tau in FILTER_TAUS:
                    alpha = float(np.exp(-1.0 / float(tau)))
                    ema = np.empty_like(arr, dtype=np.float32)
                    ema[0] = arr[0]
                    for i in range(1, len(arr)):
                        ema[i] = alpha * ema[i - 1] + (1.0 - alpha) * arr[i]
                    f[f"{col}_ema{tau}"] = ema
                    if tau in (50, 200, 800):
                        f[f"{col}_dev_ema{tau}"] = arr - ema
            out[split].append(f)
    return out


def _ema_causal(x: np.ndarray, cutoff_hz: float, dt_sec: float = 1.0) -> np.ndarray:
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


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, norm_kind: str = "channel"):
        super().__init__()
        self.left_pad = int((kernel_size - 1) * dilation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        if norm_kind == "group":
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
        elif norm_kind == "channel":
            self.norm1 = ChannelLayerNorm1d(channels)
            self.norm2 = ChannelLayerNorm1d(channels)
        else:
            raise ValueError(f"Unknown norm_kind={norm_kind}")
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        residual = x
        y = F.pad(x, (self.left_pad, 0))
        y = self.conv1(y)
        y = self.dropout(F.silu(self.norm1(y)))
        y = F.pad(y, (self.left_pad, 0))
        y = self.conv2(y)
        y = self.dropout(F.silu(self.norm2(y)))
        return residual + y


class DeepNoLeakTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 96,
        layers: int = 6,
        kernel_size: int = 5,
        norm_kind: str = "channel",
        dropout: float = 0.08,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        blocks = []
        for i in range(int(layers)):
            blocks.append(CausalConvBlock(hidden_size, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout, norm_kind=norm_kind))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_size, 1, kernel_size=1),
        )

    def encode_sequence(self, x):
        y = x.transpose(1, 2)
        y = self.input_proj(y)
        return self.blocks(y)

    def forward_sequence(self, x):
        y = self.encode_sequence(x)
        return torch.sigmoid(self.head(y).transpose(1, 2))

    def forward(self, x):
        return self.forward_sequence(x)[:, -1, :]


class DeepNoLeakTempAffineTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int = 96,
        layers: int = 6,
        kernel_size: int = 5,
        norm_kind: str = "channel",
        dropout: float = 0.08,
    ):
        super().__init__()
        if "T" not in feature_cols:
            raise ValueError("Temp-affine TCN requires T in feature columns")
        self.feature_cols = list(feature_cols)
        self.t_index = self.feature_cols.index("T")
        self.backbone = DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            norm_kind=norm_kind,
            dropout=dropout,
        )
        temp_hidden = max(16, hidden_size // 4)
        self.temp_affine = nn.Sequential(
            nn.Linear(1, temp_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(temp_hidden, 2),
        )
        nn.init.zeros_(self.temp_affine[-1].weight)
        nn.init.zeros_(self.temp_affine[-1].bias)

    def _base_logits(self, x):
        y = self.backbone.encode_sequence(x)
        return self.backbone.head(y).transpose(1, 2)

    def forward_sequence(self, x):
        logits = self._base_logits(x)
        affine = self.temp_affine(x[..., self.t_index:self.t_index + 1])
        scale = 1.0 + 0.25 * torch.tanh(affine[..., 0:1])
        bias = 0.5 * torch.tanh(affine[..., 1:2])
        return torch.sigmoid(logits * scale + bias)

    def forward(self, x):
        return self.forward_sequence(x)[:, -1, :]


class DeepNoLeakPredBiasTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int = 96,
        layers: int = 6,
        kernel_size: int = 5,
        norm_kind: str = "channel",
        dropout: float = 0.08,
        logit_limit: float = 1.5,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        gate_names = [
            "T",
            "V_raw",
            "V_corr_raw",
            "I_raw",
            "absI",
            "dI",
            "R0",
            "V_pol_raw",
            "V_hys_raw",
            "V_ohm_raw",
            "V_pol_fast_raw",
            "V_pol_mid_raw",
            "V_pol_slow_raw",
            "V_residual_low",
            "V_residual_mid",
            "V_residual_high",
        ]
        self.gate_indices = [self.feature_cols.index(c) for c in gate_names if c in self.feature_cols]
        if not self.gate_indices:
            raise ValueError("Prediction-bias TCN requires at least one calibration input feature")
        self.logit_limit = float(logit_limit)
        self.backbone = DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            norm_kind=norm_kind,
            dropout=dropout,
        )
        cal_hidden = max(32, hidden_size // 2)
        self.calibrator = nn.Sequential(
            nn.Linear(len(self.gate_indices) + 1, cal_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(cal_hidden, cal_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(cal_hidden, 1),
        )
        nn.init.zeros_(self.calibrator[-1].weight)
        nn.init.zeros_(self.calibrator[-1].bias)

    def _base_logits(self, x):
        y = self.backbone.encode_sequence(x)
        return self.backbone.head(y).transpose(1, 2)

    def forward_sequence(self, x):
        logits = self._base_logits(x)
        base_p = torch.sigmoid(logits)
        cal_in = torch.cat([base_p.detach(), x[..., self.gate_indices]], dim=-1)
        delta = self.logit_limit * torch.tanh(self.calibrator(cal_in))
        return torch.sigmoid(logits + delta)

    def forward(self, x):
        return self.forward_sequence(x)[:, -1, :]


class DeepNoLeakMoETCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int = 96,
        layers: int = 5,
        kernel_size: int = 5,
        norm_kind: str = "channel",
        dropout: float = 0.04,
        num_experts: int = 4,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        gate_names = ["T", "V_raw", "V_corr_raw", "I_raw", "absI", "dI", "R0"]
        self.gate_indices = [self.feature_cols.index(c) for c in gate_names if c in self.feature_cols]
        if not self.gate_indices:
            raise ValueError("MoE TCN requires at least one gate input feature")
        self.backbone = DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            norm_kind=norm_kind,
            dropout=dropout,
        )
        self.expert_head = nn.Conv1d(hidden_size, int(num_experts), kernel_size=1)
        gate_hidden = max(16, hidden_size // 2)
        self.gate = nn.Sequential(
            nn.Linear(len(self.gate_indices), gate_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_hidden, int(num_experts)),
        )

    def forward_sequence(self, x):
        h = self.backbone.encode_sequence(x).transpose(1, 2)
        expert_logits = self.expert_head(h.transpose(1, 2)).transpose(1, 2)
        gates = torch.softmax(self.gate(x[..., self.gate_indices]), dim=-1)
        logits = torch.sum(expert_logits * gates, dim=-1, keepdim=True)
        return torch.sigmoid(logits)

    def forward(self, x):
        return self.forward_sequence(x)[:, -1, :]


class DeepNoLeakGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.05):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
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

    def forward_sequence(self, x):
        z = self.input_proj(x)
        out, _ = self.gru(z)
        return torch.sigmoid(self.head(self.norm(out)))

    def forward(self, x):
        return self.forward_sequence(x)[:, -1, :]


class DeepNoLeakLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.05):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.lstm = nn.LSTM(
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

    def forward_sequence(self, x):
        z = self.input_proj(x)
        out, _ = self.lstm(z)
        return torch.sigmoid(self.head(self.norm(out)))

    def forward(self, x):
        return self.forward_sequence(x)[:, -1, :]


class SequenceWindowDataset(DecomposedWindowDataset):
    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        x = f["x"][start:end + 1]
        if self.target_label != "physical":
            raise ValueError("SequenceWindowDataset currently supports physical target only")
        y = f["y_physical"][start:end + 1, None]
        soc_for_bin = float(f["y_physical"][end])
        if soc_for_bin < 0.2:
            soc_bin = 0
        elif soc_for_bin <= 0.8:
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
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


def _window_local_residual_bands(x: torch.Tensor, feature_cols: list[str] | None) -> torch.Tensor:
    if feature_cols is None:
        raise ValueError("feature_cols are required for local residual window features")
    required = ["V_raw", "V_corr_raw", "T", "absI", "dI"]
    missing = [c for c in required if c not in feature_cols]
    if missing:
        raise ValueError(f"local residual window features require columns: {missing}")
    v_raw = x[:, feature_cols.index("V_raw")]
    v_corr = x[:, feature_cols.index("V_corr_raw")]
    temp = x[:, feature_cols.index("T")]
    abs_i = x[:, feature_cols.index("absI")]
    abs_di = torch.abs(x[:, feature_cols.index("dI")])
    residual = v_raw - v_corr

    def ema(series: torch.Tensor, cutoff_hz: float) -> torch.Tensor:
        alpha = float(np.exp(-2.0 * np.pi * float(cutoff_hz)))
        alpha = min(max(alpha, 0.0), 0.999999)
        out = torch.empty_like(series)
        out[0] = series[0]
        for i in range(1, int(series.shape[0])):
            out[i] = alpha * out[i - 1] + (1.0 - alpha) * series[i]
        return out

    low = ema(residual, 0.003)
    mid_lp = ema(residual, 0.03)
    mid = mid_lp - low
    high = residual - mid_lp
    return torch.stack(
        [
            residual,
            low,
            mid,
            high,
            low * temp,
            mid * abs_i,
            high * abs_di,
        ],
        dim=1,
    )


def augment_window_tensor(x: torch.Tensor, mode: str, feature_cols: list[str] | None = None) -> torch.Tensor:
    if mode == "raw":
        return x
    x = x.to(dtype=torch.float32)
    parts = [x]
    if mode in {"delta_start", "delta_start_time", "delta_start_time_local_residual"}:
        parts.append(x - x[:1])
    else:
        raise ValueError(f"Unknown window_feature_mode={mode}")
    if mode in {"delta_start_time", "delta_start_time_local_residual"}:
        # This is the normalized position inside the current window, not an absolute timestep.
        t = torch.linspace(0.0, 1.0, x.shape[0], dtype=x.dtype).view(-1, 1)
        parts.append(t)
    if mode == "delta_start_time_local_residual":
        parts.append(_window_local_residual_bands(x, feature_cols))
    return torch.cat(parts, dim=1)


def augmented_input_dim(input_dim: int, mode: str) -> int:
    if mode == "raw":
        return int(input_dim)
    if mode == "delta_start":
        return int(input_dim) * 2
    if mode == "delta_start_time":
        return int(input_dim) * 2 + 1
    if mode == "delta_start_time_local_residual":
        return int(input_dim) * 2 + 1 + 7
    raise ValueError(f"Unknown window_feature_mode={mode}")


class AugmentedWindowDataset(DecomposedWindowDataset):
    def __init__(self, *args, window_feature_mode: str = "raw", **kwargs):
        super().__init__(*args, **kwargs)
        self.window_feature_mode = str(window_feature_mode)

    def __getitem__(self, idx):
        x, y, meta = super().__getitem__(idx)
        return augment_window_tensor(x, self.window_feature_mode, self.feature_cols), y, meta


class AugmentedSequenceWindowDataset(SequenceWindowDataset):
    def __init__(self, *args, window_feature_mode: str = "raw", **kwargs):
        super().__init__(*args, **kwargs)
        self.window_feature_mode = str(window_feature_mode)

    def __getitem__(self, idx):
        x, y, meta = super().__getitem__(idx)
        return augment_window_tensor(x, self.window_feature_mode, self.feature_cols), y, meta


def make_model(cfg: DeepNoLeakConfig, input_dim: int, feature_cols: list[str] | None = None):
    if cfg.model_kind == "gru":
        return DeepNoLeakGRU(
            input_dim=input_dim,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            dropout=cfg.dropout,
        )
    if cfg.model_kind == "tcn":
        return DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            kernel_size=cfg.kernel_size,
            norm_kind=cfg.norm_kind,
            dropout=cfg.dropout,
        )
    if cfg.model_kind == "tcn_temp_affine":
        if feature_cols is None:
            raise ValueError("feature_cols are required for tcn_temp_affine")
        return DeepNoLeakTempAffineTCN(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            kernel_size=cfg.kernel_size,
            norm_kind=cfg.norm_kind,
            dropout=cfg.dropout,
        )
    if cfg.model_kind == "tcn_pred_bias":
        if feature_cols is None:
            raise ValueError("feature_cols are required for tcn_pred_bias")
        return DeepNoLeakPredBiasTCN(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            kernel_size=cfg.kernel_size,
            norm_kind=cfg.norm_kind,
            dropout=cfg.dropout,
        )
    if cfg.model_kind == "tcn_moe":
        if feature_cols is None:
            raise ValueError("feature_cols are required for tcn_moe")
        return DeepNoLeakMoETCN(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            kernel_size=cfg.kernel_size,
            norm_kind=cfg.norm_kind,
            dropout=cfg.dropout,
            num_experts=cfg.num_experts,
        )
    if cfg.model_kind == "lstm":
        return DeepNoLeakLSTM(
            input_dim=input_dim,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model_kind={cfg.model_kind}")


def frame_to_tensors(frame: pd.DataFrame, cols: list[str]):
    x = torch.as_tensor(frame[cols].to_numpy(np.float32)[None, :, :], device=device)
    y = torch.as_tensor(frame["SOC_physical"].to_numpy(np.float32)[None, :, None], device=device)
    return x, y


def pointwise_loss(pred: torch.Tensor, y: torch.Tensor, loss_kind: str, huber_beta: float):
    if loss_kind == "mae":
        return torch.abs(pred - y)
    if loss_kind == "mse":
        return (pred - y) ** 2
    if loss_kind == "huber":
        return F.smooth_l1_loss(pred, y, beta=float(huber_beta), reduction="none")
    raise ValueError(f"Unknown loss_kind={loss_kind}")


def sequence_loss_by_temperature(
    model,
    frames: list[pd.DataFrame],
    cols: list[str],
    lambda_rex: float,
    lambda_smooth: float,
    loss_kind: str,
    huber_beta: float,
):
    temp_losses = {}
    frame_losses = []
    abs_losses = []
    smooth_losses = []
    for frame in frames:
        x, y = frame_to_tensors(frame, cols)
        pred = model.forward_sequence(x)
        abs_loss = torch.mean(torch.abs(pred - y))
        loss = torch.mean(pointwise_loss(pred, y, loss_kind, huber_beta))
        if float(lambda_smooth) > 0.0 and pred.size(1) > 1:
            smooth_loss = torch.mean(torch.abs((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])))
            loss = loss + float(lambda_smooth) * smooth_loss
        else:
            smooth_loss = loss.new_tensor(0.0)
        temp = float(frame["temperature"].iloc[0])
        temp_losses.setdefault(temp, []).append(loss)
        frame_losses.append(loss)
        abs_losses.append(abs_loss)
        smooth_losses.append(smooth_loss)
    per_temp = [torch.stack(v).mean() for _, v in sorted(temp_losses.items())]
    stack = torch.stack(per_temp) if per_temp else torch.stack(frame_losses)
    mean_loss = stack.mean()
    rex_loss = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
    return mean_loss + float(lambda_rex) * rex_loss, mean_loss, rex_loss, torch.stack(abs_losses).mean(), torch.stack(smooth_losses).mean(), {
        float(k): float(torch.stack(v).mean().detach().cpu()) for k, v in temp_losses.items()
    }


@torch.no_grad()
def predict_trajectories(model, frames: list[pd.DataFrame], cols: list[str], model_name: str):
    rows = []
    model.eval()
    for frame in frames:
        f = frame.reset_index(drop=True)
        x, _ = frame_to_tensors(f, cols)
        pred = model.forward_sequence(x).detach().cpu().numpy()[0, :, 0]
        df = pd.DataFrame({
            "model_name": model_name,
            "target_label": "physical",
            "trajectory_id": f["trajectory_id"].to_numpy(),
            "file_name": f["file_name"].to_numpy() if "file_name" in f.columns else f["trajectory_id"].to_numpy(),
            "drive_cycle": f["drive_cycle"].to_numpy(),
            "temperature": f["temperature"].to_numpy(),
            "end_index": f["end_index"].to_numpy(),
            "y_true": f["SOC_physical"].to_numpy(np.float32),
            "y_pred": pred,
        })
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = np.abs(df["error"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _move(x):
    return x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def make_eval_loader(ds, cfg: DeepNoLeakConfig):
    num_workers = int(cfg.num_workers)
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _meta_to_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def _rex_group_keys(meta, group_name: str):
    temps = [float(v) for v in _meta_to_list(meta["temperature"])]
    drives = [str(v) for v in _meta_to_list(meta["drive_cycle"])]
    soc_bins = [int(v) for v in _meta_to_list(meta["soc_bin"])]
    if group_name == "temperature":
        return [f"T{t:g}" for t in temps]
    if group_name == "drive":
        return [f"D{d}" for d in drives]
    if group_name == "temperature_drive":
        return [f"T{t:g}_{d}" for t, d in zip(temps, drives)]
    if group_name == "temperature_soc":
        return [f"T{t:g}_S{s}" for t, s in zip(temps, soc_bins)]
    if group_name == "drive_soc":
        return [f"D{d}_S{s}" for d, s in zip(drives, soc_bins)]
    if group_name == "temperature_drive_soc":
        return [f"T{t:g}_{d}_S{s}" for t, d, s in zip(temps, drives, soc_bins)]
    raise ValueError(f"Unknown rex_group={group_name}")


def temp_balanced_rex_loss(
    pred,
    y,
    meta,
    lambda_rex: float,
    rex_group: str,
    loss_kind: str,
    huber_beta: float,
    lambda_smooth: float,
    endpoint_loss_weight: float,
    lambda_worst: float,
):
    if pred.ndim == 3:
        sample_loss = torch.mean(pointwise_loss(pred, y, loss_kind, huber_beta), dim=(1, 2))
        if float(endpoint_loss_weight) > 0.0:
            endpoint_loss = torch.mean(pointwise_loss(pred[:, -1, :], y[:, -1, :], loss_kind, huber_beta), dim=1)
            sample_loss = sample_loss + float(endpoint_loss_weight) * endpoint_loss
        if float(lambda_smooth) > 0.0 and pred.size(1) > 1:
            smooth_loss = torch.mean(torch.abs((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])))
        else:
            smooth_loss = pred.new_tensor(0.0)
    else:
        sample_loss = torch.mean(pointwise_loss(pred, y, loss_kind, huber_beta), dim=1)
        smooth_loss = pred.new_tensor(0.0)
    group_keys = _rex_group_keys(meta, rex_group)
    losses = []
    by_group = {}
    for key in sorted(set(group_keys)):
        idx = [i for i, k in enumerate(group_keys) if k == key]
        if not idx:
            continue
        mask = torch.as_tensor(idx, device=pred.device, dtype=torch.long)
        lt = sample_loss.index_select(0, mask).mean()
        losses.append(lt)
        by_group[key] = float(lt.detach().cpu())
    stack = torch.stack(losses) if losses else sample_loss.mean().view(1)
    mean_loss = stack.mean()
    rex_loss = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
    worst_loss = stack.max() if len(stack) > 1 else stack.new_tensor(0.0)
    total = mean_loss + float(lambda_rex) * rex_loss + float(lambda_worst) * worst_loss + float(lambda_smooth) * smooth_loss
    return total, mean_loss, rex_loss, smooth_loss, by_group


@torch.no_grad()
def predict(model, loader) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        pred = model(_move(x)).detach().cpu().numpy()[:, 0]
        yy = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = "physical"
        mdf["y_true"] = yy
        mdf["y_pred"] = pred
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def run_deep_no_leak(cfg: DeepNoLeakConfig):
    cfg.base_dir = Path(cfg.base_dir)
    configure_torch_runtime()
    if torch.cuda.is_available() and cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(cfg.seed)
    base_cfg = make_cfg()
    base_cfg.output_dir = cfg.base_dir
    base_cfg.base_dir = cfg.base_dir
    base_cfg = experiment_cfg(base_cfg, cfg.experiment)
    if cfg.feature_dir:
        base_cfg.decomposed_dir = cfg.base_dir / cfg.feature_dir
    configure_strict_training(base_cfg)
    base_cfg.window_len = int(cfg.window_len)
    base_cfg.stride = int(cfg.stride)
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.lstm_lr = float(cfg.lr)
    base_cfg.lstm_weight_decay = float(cfg.weight_decay)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    lookup = load_smoothq_lookup(cfg.base_dir)
    frames = add_derived_features(load_relabelled_frames(base_cfg, cfg.experiment, lookup))
    cols = feature_columns(cfg.feature_set)
    available = set().union(*(set(f.columns) for split in frames.values() for f in split))
    missing = [c for c in cols if c not in available]
    if missing:
        raise KeyError(f"Missing selected feature columns: {missing}")
    scaled, _ = make_scaled_frames_for_ablation(frames, cols)
    input_dim = augmented_input_dim(len(cols), cfg.window_feature_mode)
    if cfg.trajectory_mode:
        if cfg.window_feature_mode != "raw":
            raise ValueError("window_feature_mode is only supported for fixed-window stateless training")
        model_name = f"DeepNoLeakTraj_{cfg.model_kind}_{cfg.feature_set}_seed{cfg.seed}"
        model = make_model(cfg, len(cols), cols).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        history = []
        for ep in range(1, int(cfg.epochs) + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, mean_loss, rex_loss, abs_loss, smooth_loss, by_temp = sequence_loss_by_temperature(
                model,
                scaled["train"],
                cols,
                cfg.lambda_rex,
                cfg.lambda_smooth,
                cfg.loss_kind,
                cfg.huber_beta,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            row = {
                "model_name": model_name,
                "experiment": cfg.experiment,
                "epoch": ep,
                "loss": float(loss.detach().cpu()),
                "mean_abs_loss": float(mean_loss.detach().cpu()),
                "raw_mae_loss": float(abs_loss.detach().cpu()),
                "smooth_delta_loss": float(smooth_loss.detach().cpu()),
                "rex_var": float(rex_loss.detach().cpu()),
            }
            for k, v in by_temp.items():
                row[f"train_abs_loss_temp_{k:g}"] = float(v)
            history.append(row)
            if ep == 1 or ep == cfg.epochs or ep % max(1, int(cfg.print_every)) == 0:
                print(
                    f"{model_name} epoch={ep} loss={row['loss']:.5f} "
                    f"mean={row['mean_abs_loss']:.5f} raw_mae={row['raw_mae_loss']:.5f} "
                    f"smooth={row['smooth_delta_loss']:.5f} rex={row['rex_var']:.6f}",
                    flush=True,
                )
        pred = predict_trajectories(model, scaled["test"], cols, model_name)
        lookup_features = build_prediction_feature_lookup(frames)
        pred = attach_prediction_features(pred.assign(split="test", ablation=model_name), lookup_features, ablation_name=model_name, target_label="physical")
        pred["experiment"] = cfg.experiment
        pred["seed"] = int(cfg.seed)
        overall = _overall_metrics(pred)
        overall["experiment"] = cfg.experiment
        by_temp = variance_by_temperature(pred)
        by_temp["experiment"] = cfg.experiment
        omitted_temp = float(EXPERIMENTS[cfg.experiment]["omitted_temp_C"])
        omitted = by_temp[np.isclose(by_temp["temperature_C"].astype(float), omitted_temp)]
        seen = by_temp[~np.isclose(by_temp["temperature_C"].astype(float), omitted_temp)]
        focus = pd.DataFrame([{
            "experiment": cfg.experiment,
            "model_name": model_name,
            "omitted_temperature_C": omitted_temp,
            "omitted_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
            "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
            "overall_MAE_pct": float(overall["MAE_pct"].iloc[0]) if len(overall) else np.nan,
            "worst_temperature_MAE_pct": float(by_temp["MAE_pct"].max()) if len(by_temp) else np.nan,
        }])
        prefix = cfg.output_prefix
        pred.to_csv(cfg.base_dir / f"{prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
        pd.DataFrame(history).to_csv(cfg.base_dir / f"{prefix}_history.csv", index=False)
        overall.to_csv(cfg.base_dir / f"{prefix}_overall.csv", index=False)
        by_temp.to_csv(cfg.base_dir / f"{prefix}_by_temperature.csv", index=False)
        focus.to_csv(cfg.base_dir / f"{prefix}_focus.csv", index=False)
        metadata = {
            **cfg.__dict__,
            "base_dir": str(cfg.base_dir),
            "model_name": model_name,
            "feature_columns": cols,
            "forbidden_input_tokens": list(FORBIDDEN_INPUT_TOKENS),
            "uses_soc_input": False,
            "uses_cumulative_input": False,
            "uses_current_integration": False,
            "trajectory_mode": True,
            "train_trajectories": int(len(scaled["train"])),
            "test_trajectories": int(len(scaled["test"])),
        }
        (cfg.base_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        print("Overall:")
        print(overall.to_string(index=False))
        print("Focus:")
        print(focus.to_string(index=False))
        return {"pred": pred, "overall": overall, "by_temperature": by_temp, "focus": focus, "history": pd.DataFrame(history)}

    use_sequence_objective = bool(cfg.sequence_loss)
    if cfg.window_feature_mode == "raw":
        train_ds_cls = SequenceWindowDataset if use_sequence_objective else DecomposedWindowDataset
        train_ds = train_ds_cls(scaled["train"], cols, base_cfg.window_len, base_cfg.stride, target_label="physical")
        test_ds = DecomposedWindowDataset(scaled["test"], cols, base_cfg.window_len, 1, target_label="physical")
    else:
        train_ds_cls = AugmentedSequenceWindowDataset if use_sequence_objective else AugmentedWindowDataset
        train_ds = train_ds_cls(
            scaled["train"],
            cols,
            base_cfg.window_len,
            base_cfg.stride,
            target_label="physical",
            window_feature_mode=cfg.window_feature_mode,
        )
        test_ds = AugmentedWindowDataset(
            scaled["test"],
            cols,
            base_cfg.window_len,
            1,
            target_label="physical",
            window_feature_mode=cfg.window_feature_mode,
        )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Empty dataset train={len(train_ds)} test={len(test_ds)}")
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    test_loader = make_eval_loader(test_ds, cfg)
    model_name = f"DeepNoLeak_{cfg.model_kind}_{cfg.feature_set}_w{cfg.window_len}_s{cfg.stride}_seed{cfg.seed}"
    model = make_model(cfg, input_dim, cols).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        mean_losses = []
        rex_losses = []
        smooth_losses = []
        temp_loss = {}
        for x, y, meta in train_loader:
            x = _move(x)
            y = _move(y)
            if use_sequence_objective:
                pred = model.forward_sequence(x)
            else:
                pred = model(x)
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
            mean_losses.append(float(mean_loss.detach().cpu()))
            rex_losses.append(float(rex_loss.detach().cpu()))
            smooth_losses.append(float(smooth_loss.detach().cpu()))
            for k, v in by_group.items():
                temp_loss.setdefault(k, []).append(v)
        row = {
            "model_name": model_name,
            "experiment": cfg.experiment,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_abs_loss": float(np.mean(mean_losses)),
            "rex_var": float(np.mean(rex_losses)),
            "smooth_delta_loss": float(np.mean(smooth_losses)),
        }
        for k, v in temp_loss.items():
            safe_key = str(k).replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe_key}"] = float(np.mean(v))
        history.append(row)
        if ep == 1 or ep == cfg.epochs or ep % max(1, int(cfg.print_every)) == 0:
            print(
                f"{model_name} epoch={ep} loss={row['loss']:.5f} "
                f"mean={row['mean_abs_loss']:.5f} smooth={row['smooth_delta_loss']:.5f} "
                f"rex={row['rex_var']:.6f}",
                flush=True,
            )
    pred = predict(model, test_loader)
    lookup_features = build_prediction_feature_lookup(frames)
    pred = attach_prediction_features(pred.assign(split="test", ablation=model_name), lookup_features, ablation_name=model_name, target_label="physical")
    pred["experiment"] = cfg.experiment
    pred["seed"] = int(cfg.seed)
    overall = _overall_metrics(pred)
    overall["experiment"] = cfg.experiment
    by_temp = variance_by_temperature(pred)
    by_temp["experiment"] = cfg.experiment
    omitted_temp = float(EXPERIMENTS[cfg.experiment]["omitted_temp_C"])
    omitted = by_temp[np.isclose(by_temp["temperature_C"].astype(float), omitted_temp)]
    seen = by_temp[~np.isclose(by_temp["temperature_C"].astype(float), omitted_temp)]
    focus = pd.DataFrame([{
        "experiment": cfg.experiment,
        "model_name": model_name,
        "omitted_temperature_C": omitted_temp,
        "omitted_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
        "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
        "overall_MAE_pct": float(overall["MAE_pct"].iloc[0]) if len(overall) else np.nan,
        "worst_temperature_MAE_pct": float(by_temp["MAE_pct"].max()) if len(by_temp) else np.nan,
    }])
    prefix = cfg.output_prefix
    pred.to_csv(cfg.base_dir / f"{prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    pd.DataFrame(history).to_csv(cfg.base_dir / f"{prefix}_history.csv", index=False)
    overall.to_csv(cfg.base_dir / f"{prefix}_overall.csv", index=False)
    by_temp.to_csv(cfg.base_dir / f"{prefix}_by_temperature.csv", index=False)
    focus.to_csv(cfg.base_dir / f"{prefix}_focus.csv", index=False)
    metadata = {
        **cfg.__dict__,
        "base_dir": str(cfg.base_dir),
        "model_name": model_name,
        "feature_columns": cols,
        "input_feature_dim": int(input_dim),
        "forbidden_input_tokens": list(FORBIDDEN_INPUT_TOKENS),
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_current_integration": False,
        "sequence_loss": bool(cfg.sequence_loss),
        "sequence_loss_used": bool(use_sequence_objective),
        "train_windows": int(len(train_ds)),
        "test_windows": int(len(test_ds)),
    }
    (cfg.base_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Overall:")
    print(overall.to_string(index=False))
    print("Focus:")
    print(focus.to_string(index=False))
    return {"pred": pred, "overall": overall, "by_temperature": by_temp, "focus": focus, "history": pd.DataFrame(history)}


def parse_args():
    p = argparse.ArgumentParser(description="Deep learning-only SOC experiment without SOC/cumulative inputs or explicit current integration.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output-prefix", default="deep_no_leak")
    p.add_argument("--experiment", default="Omit N10")
    p.add_argument("--feature-dir", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model-kind",
        choices=["tcn", "tcn_temp_affine", "tcn_pred_bias", "tcn_moe", "lstm", "gru"],
        default="tcn",
    )
    p.add_argument(
        "--feature-set",
        choices=["base", "extended", "bands", "branch", "branch_bands", "spectral_bands", "voltage_only", "filtered"],
        default="extended",
    )
    p.add_argument(
        "--window-feature-mode",
        choices=["raw", "delta_start", "delta_start_time", "delta_start_time_local_residual"],
        default="raw",
    )
    p.add_argument("--window-len", type=int, default=300)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=96)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--norm-kind", choices=["channel", "group"], default="channel")
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--lambda-rex", type=float, default=0.4)
    p.add_argument(
        "--rex-group",
        choices=["temperature", "drive", "temperature_drive", "temperature_soc", "drive_soc", "temperature_drive_soc"],
        default="temperature",
    )
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    p.add_argument("--lambda-worst", type=float, default=0.0)
    p.add_argument("--loss-kind", choices=["mae", "huber", "mse"], default="mae")
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--sequence-loss", action="store_true")
    p.add_argument("--trajectory-mode", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DeepNoLeakConfig(
        base_dir=Path(args.base_dir),
        output_prefix=args.output_prefix,
        experiment=args.experiment,
        feature_dir=args.feature_dir,
        seed=args.seed,
        model_kind=args.model_kind,
        feature_set=args.feature_set,
        window_feature_mode=args.window_feature_mode,
        window_len=args.window_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        layers=args.layers,
        kernel_size=args.kernel_size,
        num_experts=args.num_experts,
        norm_kind=args.norm_kind,
        dropout=args.dropout,
        lambda_rex=args.lambda_rex,
        rex_group=args.rex_group,
        lambda_smooth=args.lambda_smooth,
        endpoint_loss_weight=args.endpoint_loss_weight,
        lambda_worst=args.lambda_worst,
        loss_kind=args.loss_kind,
        huber_beta=args.huber_beta,
        sequence_loss=bool(args.sequence_loss),
        trajectory_mode=bool(args.trajectory_mode),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        allow_tf32=bool(args.allow_tf32),
        print_every=args.print_every,
    )
    run_deep_no_leak(cfg)


if __name__ == "__main__":
    main()
