from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .data import FeatureStandardizer
from .models import collate_meta_to_frame
from .temp_variant_runner import (
    add_temperature_rbf_features,
    load_feature_frame_dict_from_csv,
)
from .training import (
    attach_prediction_features,
    build_prediction_feature_lookup,
    plot_component_gate_summary,
    summarize_component_gates,
)

try:
    from IPython.display import display
except Exception:
    display = print


R5_FEATURES = ["V_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]
R5_GATED_FEATURES = ["V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]
R5_BOUNDED_FEATURES = ["V_raw", "V_corr_raw", "I_raw", "T", "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"]
INTERACTION_COLS = ["R0_x_V_pol", "T_x_V_pol", "R0_x_absI", "V_pol_x_abs_dI"]
COMPONENT_COLS = ["V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"] + INTERACTION_COLS
HYBRID_SUMMARY_COLS = [
    "V_pol_fast_last", "V_pol_mid_last", "V_pol_slow_last",
    "V_pol_fast_mean", "V_pol_mid_mean", "V_pol_slow_mean",
    "V_pol_fast_rms", "V_pol_mid_rms", "V_pol_slow_rms",
    "V_pol_mean", "V_pol_rms", "V_pol_hf_energy",
    "V_hys_last", "V_hys_mean", "V_hys_rms", "V_hys_hf_energy",
    "R0_mean", "R0_slope", "V_ohm_rms",
]
HYBRID_RAW_COLS = [
    "V_raw", "V_corr_raw", "I_raw", "T", "dI", "absI",
    "V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0",
]


def add_interaction_features(feature_frames):
    out = {}
    for split, frames in feature_frames.items():
        out[split] = []
        for frame in frames:
            f = frame.copy()
            abs_i = f["absI"].astype(np.float32) if "absI" in f.columns else np.abs(f["I_raw"].astype(np.float32))
            abs_di = np.abs(f["dI"].astype(np.float32))
            f["R0_x_V_pol"] = f["R0"].astype(np.float32) * f["V_pol_raw"].astype(np.float32)
            f["T_x_V_pol"] = f["T"].astype(np.float32) * f["V_pol_raw"].astype(np.float32)
            f["R0_x_absI"] = f["R0"].astype(np.float32) * abs_i
            f["V_pol_x_abs_dI"] = f["V_pol_raw"].astype(np.float32) * abs_di
            out[split].append(f)
    return out


class SeqWindowDataset(Dataset):
    def __init__(self, frames, feature_cols, window_len, stride, target_col="SOC_physical"):
        self.frames = []
        self.feature_cols = list(feature_cols)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.index = []
        for fi, frame in enumerate(frames):
            frame = frame.reset_index(drop=True)
            cache = {
                "x": np.ascontiguousarray(frame[self.feature_cols].to_numpy(np.float32)),
                "y": np.ascontiguousarray(frame[target_col].to_numpy(np.float32)),
                "file_name": frame["file_name"].to_numpy(),
                "trajectory_id": frame["trajectory_id"].to_numpy(),
                "end_index": frame["end_index"].to_numpy(np.int64),
                "temperature": frame["temperature"].to_numpy(np.float32),
                "drive_cycle": frame["drive_cycle"].to_numpy(),
            }
            self.frames.append(cache)
            n = len(frame)
            if n < self.window_len:
                continue
            for start in range(0, n - self.window_len + 1, self.stride):
                end = start + self.window_len - 1
                y_seq = cache["y"][start:end + 1]
                if np.isfinite(y_seq).all():
                    self.index.append((fi, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        meta = {
            "file_name": f["file_name"][end],
            "trajectory_id": f["trajectory_id"][end],
            "end_index": int(f["end_index"][end]),
            "temperature": float(f["temperature"][end]),
            "drive_cycle": f["drive_cycle"][end],
        }
        x = f["x"][start:end + 1]
        y = f["y"][start:end + 1, None]
        return torch.from_numpy(x), torch.from_numpy(y), meta


class SeqLSTM_SOC(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def encode_sequence(self, x):
        z = self.input_proj(x)
        out, _ = self.lstm(z)
        return self.norm(out + z)

    def forward_seq(self, x):
        return torch.sigmoid(self.head(self.encode_sequence(x)))

    def forward(self, x):
        return self.forward_seq(x)[:, -1, :]


class GatedSeqLSTM_SOC(SeqLSTM_SOC):
    def __init__(self, input_dim, feature_cols, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__(input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.feature_cols = list(feature_cols)
        self.component_names = ["V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        missing = [c for c in self.component_names if c not in self.feature_cols]
        if missing:
            raise ValueError(f"Gated model requires component columns: {missing}")
        gate_input_names = [
            "T", "V_raw", "V_corr_raw", "absI", "dI", "R0",
            "V_pol_raw", "V_hys_raw", "V_ohm_raw",
            "R0_x_V_pol", "T_x_V_pol", "R0_x_absI", "V_pol_x_abs_dI",
        ]
        self.gate_input_names = [c for c in gate_input_names if c in self.feature_cols]
        self.gate_input_indices = [self.feature_cols.index(c) for c in self.gate_input_names]
        self.component_indices = [self.feature_cols.index(c) for c in self.component_names]
        self.component_gate = nn.Sequential(
            nn.Linear(len(self.gate_input_indices), 32), nn.SiLU(),
            nn.Linear(32, 3), nn.Sigmoid()
        )

    def component_gates(self, x):
        return self.component_gate(x[..., self.gate_input_indices])

    def apply_component_gates(self, x):
        gates = self.component_gates(x)
        gated_by_index = {idx: j for j, idx in enumerate(self.component_indices)}
        cols = []
        for idx in range(x.size(-1)):
            col = x[..., idx:idx + 1]
            if idx in gated_by_index:
                col = col * gates[..., gated_by_index[idx]:gated_by_index[idx] + 1]
            cols.append(col)
        return torch.cat(cols, dim=-1), gates

    def forward_seq(self, x):
        x_eff, _ = self.apply_component_gates(x)
        return torch.sigmoid(self.head(self.encode_sequence(x_eff)))


class BoundedSeqSOC(nn.Module):
    def __init__(
        self,
        feature_cols,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        correction_limit=0.05,
        gated=False,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.correction_limit = float(correction_limit)
        self.gated = bool(gated)
        self.base_cols = [c for c in ["V_raw", "I_raw", "T", "V_corr_raw"] if c in self.feature_cols]
        self.comp_cols = [c for c in ["V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0", *INTERACTION_COLS] if c in self.feature_cols]
        if not self.base_cols or not self.comp_cols:
            raise ValueError("Bounded model requires base and component columns.")
        self.base_idx = [self.feature_cols.index(c) for c in self.base_cols]
        self.comp_idx = [self.feature_cols.index(c) for c in self.comp_cols]
        self.base_proj = nn.Linear(len(self.base_idx), hidden_size)
        self.base_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.base_norm = nn.LayerNorm(hidden_size)
        self.comp_proj = nn.Sequential(
            nn.Linear(len(self.comp_idx), hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.base_head = nn.Linear(hidden_size, 1)
        self.corr_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        if self.gated:
            gate_input_names = [
                "T", "V_raw", "V_corr_raw", "absI", "dI", "R0",
                "V_pol_raw", "V_hys_raw", "V_ohm_raw",
                "R0_x_V_pol", "T_x_V_pol", "R0_x_absI", "V_pol_x_abs_dI",
            ]
            self.gate_input_names = [c for c in gate_input_names if c in self.feature_cols]
            self.gate_input_indices = [self.feature_cols.index(c) for c in self.gate_input_names]
            self.gated_component_names = ["V_pol_raw", "V_hys_raw", "V_ohm_raw"]
            self.gated_component_indices = [self.feature_cols.index(c) for c in self.gated_component_names]
            self.component_gate = nn.Sequential(
                nn.Linear(len(self.gate_input_indices), 32), nn.SiLU(),
                nn.Linear(32, 3), nn.Sigmoid()
            )

    def component_gates(self, x):
        if not self.gated:
            return None
        return self.component_gate(x[..., self.gate_input_indices])

    def _component_values(self, x):
        comp = x[..., self.comp_idx]
        if not self.gated:
            return comp
        gates = self.component_gates(x)
        gate_by_name = {name: gates[..., j:j + 1] for j, name in enumerate(self.gated_component_names)}
        pieces = []
        for name in self.comp_cols:
            val = x[..., self.feature_cols.index(name):self.feature_cols.index(name) + 1]
            if name in gate_by_name:
                val = val * gate_by_name[name]
            pieces.append(val)
        return torch.cat(pieces, dim=-1)

    def forward_seq(self, x):
        xb = x[..., self.base_idx]
        z = self.base_proj(xb)
        out, _ = self.base_lstm(z)
        h_base = self.base_norm(out + z)
        h_comp = self.comp_proj(self._component_values(x))
        soc_base = torch.sigmoid(self.base_head(h_base))
        delta = self.correction_limit * torch.tanh(self.corr_head(torch.cat([h_base, h_comp], dim=-1)))
        return (soc_base + delta).clamp(0.0, 1.0)

    def forward(self, x):
        return self.forward_seq(x)[:, -1, :]


class HybridSummarySeqSOC(nn.Module):
    def __init__(
        self,
        feature_cols,
        raw_cols,
        summary_cols,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        gated=False,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.raw_cols = [c for c in raw_cols if c in self.feature_cols]
        self.summary_cols = [c for c in summary_cols if c in self.feature_cols]
        self.gated = bool(gated)
        if not self.raw_cols or not self.summary_cols:
            raise ValueError("HybridSummarySeqSOC requires raw and summary columns.")
        self.raw_idx = [self.feature_cols.index(c) for c in self.raw_cols]
        self.summary_idx = [self.feature_cols.index(c) for c in self.summary_cols]
        self.raw_proj = nn.Linear(len(self.raw_idx), hidden_size)
        self.raw_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.raw_norm = nn.LayerNorm(hidden_size)
        self.summary_mlp = nn.Sequential(
            nn.Linear(len(self.summary_idx), hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        if self.gated:
            self.component_names = ["V_pol_raw", "V_hys_raw", "V_ohm_raw"]
            missing = [c for c in self.component_names if c not in self.feature_cols]
            if missing:
                raise ValueError(f"Gated hybrid model requires component columns: {missing}")
            gate_input_names = [
                "T", "V_raw", "V_corr_raw", "absI", "dI", "R0",
                "V_pol_raw", "V_hys_raw", "V_ohm_raw",
            ]
            self.gate_input_names = [c for c in gate_input_names if c in self.feature_cols]
            self.gate_input_indices = [self.feature_cols.index(c) for c in self.gate_input_names]
            self.component_indices = [self.feature_cols.index(c) for c in self.component_names]
            self.component_gate = nn.Sequential(
                nn.Linear(len(self.gate_input_indices), 32), nn.SiLU(),
                nn.Linear(32, 3), nn.Sigmoid()
            )

    def component_gates(self, x):
        if not self.gated:
            return None
        return self.component_gate(x[..., self.gate_input_indices])

    def _raw_values(self, x):
        if not self.gated:
            return x[..., self.raw_idx]
        gates = self.component_gates(x)
        gated_by_index = {idx: j for j, idx in enumerate(self.component_indices)}
        pieces = []
        for name in self.raw_cols:
            idx = self.feature_cols.index(name)
            val = x[..., idx:idx + 1]
            if idx in gated_by_index:
                val = val * gates[..., gated_by_index[idx]:gated_by_index[idx] + 1]
            pieces.append(val)
        return torch.cat(pieces, dim=-1)

    def forward_seq(self, x):
        z = self.raw_proj(self._raw_values(x))
        raw_out, _ = self.raw_lstm(z)
        raw_h = self.raw_norm(raw_out + z)
        summary_h = self.summary_mlp(x[..., self.summary_idx])
        return torch.sigmoid(self.head(torch.cat([raw_h, summary_h], dim=-1)))

    def forward(self, x):
        return self.forward_seq(x)[:, -1, :]


def variance_model_spec(name):
    specs = {
        "R5_SEQ": {
            "features": R5_FEATURES,
            "kind": "seq",
            "lambda_seq": 0.5,
        },
        "R5_GATED_SEQ": {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_DERIV": {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_DERIV_OVERLAP": {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_overlap": 0.1,
            "lambda_gate_smooth": 0.03,
            "ordered_loader": True,
        },
        "R5_GATED_BOUNDED": {
            "features": R5_BOUNDED_FEATURES,
            "kind": "bounded",
            "correction_limit": 0.05,
            "gated": True,
        },
        "R5_GATED_SEQ_BOUNDED": {
            "features": R5_BOUNDED_FEATURES,
            "kind": "bounded",
            "correction_limit": 0.05,
            "gated": True,
            "lambda_seq": 0.5,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_BOUNDED_AUG": {
            "features": R5_BOUNDED_FEATURES,
            "kind": "bounded",
            "correction_limit": 0.05,
            "gated": True,
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_aug": 0.05,
            "component_noise_std": 0.01,
            "component_dropout_p": 0.1,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_R0_x_V_pol": {
            "features": R5_GATED_FEATURES + ["R0_x_V_pol"],
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_T_x_V_pol": {
            "features": R5_GATED_FEATURES + ["T_x_V_pol"],
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_R0_x_absI": {
            "features": R5_GATED_FEATURES + ["R0_x_absI"],
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_gate_smooth": 0.03,
        },
        "R5_GATED_SEQ_V_pol_x_abs_dI": {
            "features": R5_GATED_FEATURES + ["V_pol_x_abs_dI"],
            "kind": "gated_seq",
            "lambda_seq": 0.5,
            "lambda_d1": 0.1,
            "lambda_d2": 0.03,
            "lambda_gate_smooth": 0.03,
        },
    }
    return specs[name]


def make_variance_model(spec, cfg):
    if spec["kind"] == "seq":
        return SeqLSTM_SOC(
            input_dim=len(spec["features"]),
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
        )
    if spec["kind"] == "gated_seq":
        return GatedSeqLSTM_SOC(
            input_dim=len(spec["features"]),
            feature_cols=spec["features"],
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
        )
    if spec["kind"] == "bounded":
        return BoundedSeqSOC(
            feature_cols=spec["features"],
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
            correction_limit=spec.get("correction_limit", 0.05),
            gated=spec.get("gated", False),
        )
    if spec["kind"] == "hybrid_summary":
        return HybridSummarySeqSOC(
            feature_cols=spec["features"],
            raw_cols=spec.get("raw_cols", HYBRID_RAW_COLS),
            summary_cols=spec.get("summary_cols", HYBRID_SUMMARY_COLS),
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
            gated=spec.get("gated", False),
        )
    raise ValueError(f"Unknown variance model kind: {spec['kind']}")


def _move(x, cfg):
    return x.to(device=device, dtype=torch.float32, non_blocking=bool(getattr(cfg, "cuda_non_blocking", True)) and device.type == "cuda")


def _loader(ds, cfg, *, shuffle):
    workers = max(0, int(getattr(cfg, "dataloader_num_workers", 0)))
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": bool(getattr(cfg, "dataloader_pin_memory", True)) and device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(cfg, "dataloader_prefetch_factor", 2))
    return DataLoader(ds, **kwargs)


def _scale_frames(feature_frames, feature_cols):
    train_ids = [f["trajectory_id"].iloc[0] for f in feature_frames["train"]]
    scaler = FeatureStandardizer().fit(feature_frames["train"], feature_cols, fit_ids=train_ids)
    test_ids = {f["trajectory_id"].iloc[0] for f in feature_frames["test"]}
    assert scaler.fit_ids.isdisjoint(test_ids), "Scaler leakage: test IDs included in fit"
    return {split: [scaler.transform_frame(f) for f in frames] for split, frames in feature_frames.items()}, scaler


def derivative_losses(pred_seq, y_seq):
    d_pred = pred_seq[:, 1:, :] - pred_seq[:, :-1, :]
    d_true = y_seq[:, 1:, :] - y_seq[:, :-1, :]
    d1 = F.smooth_l1_loss(d_pred, d_true, beta=0.005)
    if d_pred.size(1) > 1:
        dd_pred = d_pred[:, 1:, :] - d_pred[:, :-1, :]
        dd_true = d_true[:, 1:, :] - d_true[:, :-1, :]
        d2 = F.smooth_l1_loss(dd_pred, dd_true, beta=0.005)
    else:
        d2 = pred_seq.new_tensor(0.0)
    return d1, d2


def overlap_consistency_loss(pred_seq, meta):
    tids = list(meta["trajectory_id"])
    end_index = meta["end_index"]
    if torch.is_tensor(end_index):
        end_index = end_index.detach().cpu().numpy().tolist()
    if len(tids) < 2:
        return pred_seq.new_tensor(0.0)
    adjacent = [
        i for i in range(len(tids) - 1)
        if tids[i] == tids[i + 1] and int(end_index[i + 1]) == int(end_index[i]) + 1
    ]
    if not adjacent:
        return pred_seq.new_tensor(0.0)
    idx = torch.as_tensor(adjacent, device=pred_seq.device, dtype=torch.long)
    return torch.mean(torch.abs(pred_seq[idx, 1:, :] - pred_seq[idx + 1, :-1, :]))


def gate_smoothness_loss(model, x):
    if not hasattr(model, "component_gates"):
        return x.new_tensor(0.0)
    gates = model.component_gates(x)
    if gates is None or gates.size(1) < 2:
        return x.new_tensor(0.0)
    return torch.mean(torch.abs(gates[:, 1:, :] - gates[:, :-1, :]))


def augment_components(x, feature_cols, spec):
    x_aug = x.clone()
    indices = [feature_cols.index(c) for c in COMPONENT_COLS if c in feature_cols]
    if not indices:
        return x_aug
    noise_std = float(spec.get("component_noise_std", 0.0))
    dropout_p = float(spec.get("component_dropout_p", 0.0))
    if noise_std > 0:
        x_aug[..., indices] = x_aug[..., indices] + torch.randn_like(x_aug[..., indices]) * noise_std
    if dropout_p > 0:
        keep = (torch.rand_like(x_aug[..., indices]) > dropout_p).float()
        x_aug[..., indices] = x_aug[..., indices] * keep
    return x_aug


def variance_loss(model, x, y_seq, meta, spec):
    pred_seq = model.forward_seq(x)
    l_last = F.l1_loss(pred_seq[:, -1, :], y_seq[:, -1, :])
    l_seq = F.l1_loss(pred_seq, y_seq)
    total = l_last + float(spec.get("lambda_seq", 0.0)) * l_seq
    parts = {"last": l_last, "seq": l_seq}
    if spec.get("lambda_d1", 0.0) or spec.get("lambda_d2", 0.0):
        d1, d2 = derivative_losses(pred_seq, y_seq)
        total = total + float(spec.get("lambda_d1", 0.0)) * d1 + float(spec.get("lambda_d2", 0.0)) * d2
        parts["d1"] = d1
        parts["d2"] = d2
    if spec.get("lambda_overlap", 0.0):
        ov = overlap_consistency_loss(pred_seq, meta)
        total = total + float(spec.get("lambda_overlap", 0.0)) * ov
        parts["overlap"] = ov
    if spec.get("lambda_gate_smooth", 0.0):
        gs = gate_smoothness_loss(model, x)
        total = total + float(spec.get("lambda_gate_smooth", 0.0)) * gs
        parts["gate_smooth"] = gs
    if spec.get("lambda_mono", 0.0):
        d_pred = pred_seq[:, 1:, :] - pred_seq[:, :-1, :]
        eps = float(spec.get("mono_eps", 0.003))
        mono = F.relu(d_pred - eps).mean()
        total = total + float(spec.get("lambda_mono", 0.0)) * mono
        parts["mono"] = mono
    if spec.get("lambda_aug", 0.0):
        x_aug = augment_components(x, spec["features"], spec)
        pred_aug = model.forward_seq(x_aug)
        aug = F.smooth_l1_loss(pred_aug, pred_seq.detach(), beta=0.01)
        total = total + float(spec.get("lambda_aug", 0.0)) * aug
        parts["aug"] = aug
    if spec.get("lambda_jac", 0.0):
        jac_batch = min(int(spec.get("jacobian_batch_size", 128)), x.size(0))
        x_jac = x[:jac_batch].detach().clone().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            pred_jac = model.forward_seq(x_jac)[:, -1, :].sum()
        grad = torch.autograd.grad(pred_jac, x_jac, create_graph=True, retain_graph=True, only_inputs=True)[0]
        component_indices = [spec["features"].index(c) for c in ["V_pol_raw", "V_hys_raw", "V_ohm_raw", "R0"] if c in spec["features"]]
        jac = grad[..., component_indices].pow(2).mean() if component_indices else x.new_tensor(0.0)
        total = total + float(spec.get("lambda_jac", 0.0)) * jac
        parts["jac"] = jac
    parts["total"] = total
    return total, parts


@torch.no_grad()
def predict_variance_model(model, loader, cfg):
    model.eval()
    rows = []
    for x, y_seq, meta in loader:
        pred_seq = model.forward_seq(_move(x, cfg)).detach().cpu().numpy()
        yy = y_seq.numpy()
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = "physical"
        mdf["y_true"] = yy[:, -1, 0]
        mdf["y_pred"] = pred_seq[:, -1, 0]
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def train_variance_model_from_spec(feature_frames, model_name, spec, cfg):
    feature_cols = spec["features"]
    scaled, scaler = _scale_frames(feature_frames, feature_cols)
    train_ds = SeqWindowDataset(scaled["train"], feature_cols, cfg.window_len, cfg.stride, target_col="SOC_physical")
    test_ds = SeqWindowDataset(scaled["test"], feature_cols, cfg.window_len, cfg.stride, target_col="SOC_physical")
    if len(train_ds) == 0:
        raise ValueError("No train windows for variance-control model.")
    model = make_variance_model(spec, cfg).to(device)
    train_loader = _loader(train_ds, cfg, shuffle=not bool(spec.get("ordered_loader", False)))
    test_loader = _loader(test_ds, cfg, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    history = []
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for x, y_seq, meta in train_loader:
            x = _move(x, cfg)
            y_seq = _move(y_seq, cfg)
            loss, _ = variance_loss(model, x, y_seq, meta, spec)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": ep, "train_loss": float(np.mean(losses))}
        history.append(row)
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 1)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"{model_name} epoch={ep} train_loss={row['train_loss']:.5f}")
    pred_test = predict_variance_model(model, test_loader, cfg)
    if spec.get("kind") == "gated_seq" or bool(spec.get("gated", False)):
        gates = summarize_component_gates(model, test_loader, "physical", model_name, cfg)
    else:
        gates = pd.DataFrame()
    return model, pd.DataFrame(history), pred_test, gates


def train_variance_model(feature_frames, model_name, cfg):
    return train_variance_model_from_spec(feature_frames, model_name, variance_model_spec(model_name), cfg)


def _overall_metrics(pred_rows):
    rows = []
    for model, g in pred_rows.groupby("model_name"):
        err = g["error"].to_numpy(np.float32)
        rows.append({
            "model_name": model,
            "n_windows": int(len(g)),
            "MAE": float(g["abs_error"].mean()),
            "MAE_pct": float(g["abs_error"].mean() * 100.0),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "error_std": float(np.std(err)),
            "error_std_pct": float(np.std(err) * 100.0),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def _trajectory_jitter_rows(pred_rows):
    rows = []
    for (model, temp, tid), g in pred_rows.sort_values("end_index").groupby(["model_name", "temperature_C", "trajectory_id"]):
        if len(g) < 4:
            continue
        yp = g["y_pred"].to_numpy(np.float64)
        yt = g["y_true"].to_numpy(np.float64)
        err = yp - yt
        dyp = np.diff(yp)
        dyt = np.diff(yt)
        ddyp = np.diff(dyp)
        ddyt = np.diff(dyt)
        pred_jitter = float(np.mean(np.abs(dyp)))
        true_jitter = float(np.mean(np.abs(dyt)))
        rows.append({
            "model_name": model,
            "temperature_C": float(temp),
            "trajectory_id": tid,
            "n_windows": int(len(g)),
            "MAE": float(np.mean(np.abs(err))),
            "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
            "error_std": float(np.std(err)),
            "error_std_pct": float(np.std(err) * 100.0),
            "pred_jitter": pred_jitter,
            "true_jitter": true_jitter,
            "jitter_ratio": float(pred_jitter / max(true_jitter, 1e-12)),
            "high_frequency_error_energy": float(np.mean(np.diff(err) ** 2)),
            "delta_soc_mae": float(np.mean(np.abs(dyp - dyt))),
            "curvature_mae": float(np.mean(np.abs(ddyp - ddyt))) if len(ddyp) else float("nan"),
        })
    return pd.DataFrame(rows)


def variance_by_temperature(pred_rows):
    jitter = _trajectory_jitter_rows(pred_rows)
    if jitter.empty:
        return jitter
    agg = (
        jitter.groupby(["model_name", "temperature_C"])[[
            "n_windows", "MAE", "MAE_pct", "RMSE", "RMSE_pct", "error_std", "error_std_pct",
            "pred_jitter", "true_jitter", "jitter_ratio", "high_frequency_error_energy",
            "delta_soc_mae", "curvature_mae",
        ]]
        .agg({
            "n_windows": "sum",
            "MAE": "mean",
            "MAE_pct": "mean",
            "RMSE": "mean",
            "RMSE_pct": "mean",
            "error_std": "mean",
            "error_std_pct": "mean",
            "pred_jitter": "mean",
            "true_jitter": "mean",
            "jitter_ratio": "mean",
            "high_frequency_error_energy": "mean",
            "delta_soc_mae": "mean",
            "curvature_mae": "mean",
        })
        .reset_index()
    )
    return agg.sort_values(["temperature_C", "MAE"]).reset_index(drop=True)


def focus_scope_metrics(pred_rows, temperatures=(0.0, 10.0, 20.0)):
    rows = []
    for (model, temp), g in pred_rows[pred_rows["temperature_C"].isin(temperatures)].groupby(["model_name", "temperature_C"]):
        scopes = {
            "overall": np.ones(len(g), dtype=bool),
            "plateau_20_80": g["is_plateau_20_80"].to_numpy(dtype=bool),
            "mid_trajectory": ((g["trajectory_fraction"] >= 1 / 3) & (g["trajectory_fraction"] <= 2 / 3)).to_numpy(dtype=bool),
            "cutoff_last10": g["is_cutoff_last10"].to_numpy(dtype=bool),
        }
        for scope, mask in scopes.items():
            gg = g.loc[mask]
            if gg.empty:
                continue
            err = gg["error"].to_numpy(np.float32)
            rows.append({
                "model_name": model,
                "temperature_C": float(temp),
                "scope": scope,
                "n_windows": int(len(gg)),
                "MAE": float(gg["abs_error"].mean()),
                "MAE_pct": float(gg["abs_error"].mean() * 100.0),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
                "error_std": float(np.std(err)),
                "error_std_pct": float(np.std(err) * 100.0),
            })
    return pd.DataFrame(rows).sort_values(["temperature_C", "scope", "MAE"]).reset_index(drop=True)


def _baseline_prediction_rows(cfg, model_names=("R5_raw_I_T_all_components", "R5_GATED"), min_endpoint=None):
    baseline_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_prediction_rows.csv"
    baseline_rows = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    if baseline_rows.empty:
        return baseline_rows
    baseline_rows = baseline_rows[baseline_rows["model_name"].isin(model_names)].copy()
    baseline_rows = baseline_rows[baseline_rows["label_type"] == "physical"].copy()
    if min_endpoint is not None and "end_index" in baseline_rows.columns:
        baseline_rows = baseline_rows[baseline_rows["end_index"] >= int(min_endpoint)].copy()
    return baseline_rows


def _load_exp_c_feature_frames(cfg):
    cfg.decomposed_dir = cfg.output_dir / "decomposed_features_train_temp_minus10_0_10_25_50"
    return load_feature_frame_dict_from_csv(cfg, decomposed_dir=cfg.decomposed_dir)


def _summary_with_temp20_focus(pred_rows):
    results = _overall_metrics(pred_rows)
    by_temp = variance_by_temperature(pred_rows)
    focus = focus_scope_metrics(pred_rows, temperatures=(0.0, 10.0, 20.0))
    if by_temp.empty:
        return results, by_temp, focus, pd.DataFrame()
    temp20 = by_temp[np.isclose(by_temp["temperature_C"].astype(float), 20.0)].copy()
    rename = {
        "MAE_pct": "temp20_MAE_pct",
        "RMSE_pct": "temp20_RMSE_pct",
        "error_std_pct": "temp20_error_std_pct",
        "pred_jitter": "temp20_pred_jitter",
        "true_jitter": "temp20_true_jitter",
        "jitter_ratio": "temp20_jitter_ratio",
        "high_frequency_error_energy": "temp20_high_frequency_error_energy",
        "delta_soc_mae": "temp20_delta_soc_mae",
        "curvature_mae": "temp20_curvature_mae",
    }
    keep = ["model_name", *rename.keys()]
    temp20_summary = temp20[[c for c in keep if c in temp20.columns]].rename(columns=rename)
    focus20 = focus[np.isclose(focus["temperature_C"].astype(float), 20.0)].copy()
    if len(focus20):
        focus_wide = (
            focus20.pivot_table(index="model_name", columns="scope", values="MAE_pct", aggfunc="first")
            .rename(columns={
                "overall": "temp20_overall_focus_MAE_pct",
                "plateau_20_80": "temp20_plateau_20_80_MAE_pct",
                "mid_trajectory": "temp20_mid_trajectory_MAE_pct",
                "cutoff_last10": "temp20_cutoff_last10_MAE_pct",
            })
            .reset_index()
        )
        temp20_summary = temp20_summary.merge(focus_wide, on="model_name", how="left")
    summary = results.merge(temp20_summary, on="model_name", how="left")
    return summary.sort_values("MAE").reset_index(drop=True), by_temp, focus, temp20


def _run_spec_grid(cfg, feature_frames, specs, *, prediction_path=None):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    pred_rows = []
    baseline_rows = _baseline_prediction_rows(cfg)
    if len(baseline_rows):
        pred_rows.append(baseline_rows)
    histories = {}
    gate_rows = []
    for model_name, spec in specs:
        print(f"\n=== sensitivity-control model: {model_name} ===")
        _, hist, pred_test, gates = train_variance_model_from_spec(feature_frames, model_name, spec, cfg)
        histories[model_name] = hist
        pred_test = pred_test.assign(split="test", ablation=model_name)
        pred_rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=model_name, target_label="physical"))
        if len(gates):
            gate_rows.append(gates)
    all_pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if prediction_path is not None:
        all_pred.to_csv(prediction_path, index=False)
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(all_pred)
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return {
        "prediction_rows": all_pred,
        "summary": summary,
        "by_temperature": by_temp,
        "focus_metrics": focus,
        "temp20_jitter": temp20,
        "histories": histories,
        "gates": gates,
    }


def plot_temp20_predictions(pred_rows, cfg):
    out_dir = cfg.output_dir / "temp20_prediction_plots"
    delta_dir = cfg.output_dir / "temp20_delta_soc_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)
    df = pred_rows[np.isclose(pred_rows["temperature_C"].astype(float), 20.0)].copy()
    if df.empty:
        return
    for model, g in df.sort_values("end_index").groupby("model_name"):
        plt.figure(figsize=(10, 3.2))
        plt.plot(g["end_index"], g["y_true"] * 100.0, label="true SOC", linewidth=1.1)
        plt.plot(g["end_index"], g["y_pred"] * 100.0, label="predicted SOC", linewidth=0.9)
        plt.xlabel("time index")
        plt.ylabel("physical SOC (%SOC)")
        plt.title(f"FUDS 20C physical SOC | {model}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}_temp20_prediction.png", dpi=180)
        plt.close()

        yp = g["y_pred"].to_numpy(np.float64)
        yt = g["y_true"].to_numpy(np.float64)
        x = g["end_index"].to_numpy()[1:]
        plt.figure(figsize=(10, 3.2))
        plt.plot(x, np.diff(yt) * 100.0, label="true delta SOC", linewidth=1.1)
        plt.plot(x, np.diff(yp) * 100.0, label="predicted delta SOC", linewidth=0.9)
        plt.xlabel("time index")
        plt.ylabel("delta SOC (%SOC/sample)")
        plt.title(f"FUDS 20C delta SOC | {model}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(delta_dir / f"{model}_temp20_delta_soc.png", dpi=180)
        plt.close()


def run_bounded_correction_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for limit in (0.03, 0.05, 0.08):
        tag = str(limit).replace("0.", "p")
        specs.append((
            f"R5_BOUNDED_lim_{tag}",
            {
                "features": R5_BOUNDED_FEATURES,
                "kind": "bounded",
                "correction_limit": limit,
                "gated": False,
            },
        ))
        specs.append((
            f"R5_GATED_BOUNDED_lim_{tag}",
            {
                "features": R5_BOUNDED_FEATURES,
                "kind": "bounded",
                "correction_limit": limit,
                "gated": True,
                "lambda_gate_smooth": 0.03,
            },
        ))
    out = _run_spec_grid(cfg, feature_frames, specs, prediction_path=cfg.output_dir / "bounded_correction_prediction_rows.csv")
    out["summary"].to_csv(cfg.output_dir / "bounded_correction_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "bounded_correction_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "bounded_correction_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_bounded_jitter_diagnostic.csv", index=False)
    print("Bounded correction summary:")
    display(out["summary"])
    print("20C bounded jitter diagnostic:")
    display(out["temp20_jitter"])
    return out


def run_component_aug_consistency_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for base_name, base_spec in (
        ("R5_GATED", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_gate_smooth": 0.03,
        }),
        ("R5_GATED_BOUNDED", {
            "features": R5_BOUNDED_FEATURES,
            "kind": "bounded",
            "correction_limit": 0.05,
            "gated": True,
            "lambda_gate_smooth": 0.03,
        }),
    ):
        for noise in (0.005, 0.01, 0.02):
            for drop in (0.05, 0.10):
                for lam in (0.03, 0.05, 0.10):
                    spec = dict(base_spec)
                    spec.update({
                        "lambda_aug": lam,
                        "component_noise_std": noise,
                        "component_dropout_p": drop,
                    })
                    name = (
                        f"{base_name}_AUG"
                        f"_n{str(noise).replace('0.', 'p')}"
                        f"_d{str(drop).replace('0.', 'p')}"
                        f"_l{str(lam).replace('0.', 'p')}"
                    )
                    specs.append((name, spec))
    out = _run_spec_grid(cfg, feature_frames, specs, prediction_path=cfg.output_dir / "component_aug_consistency_prediction_rows.csv")
    out["summary"].to_csv(cfg.output_dir / "component_aug_consistency_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "component_aug_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "component_aug_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_component_aug_jitter.csv", index=False)
    print("Component augmentation consistency summary:")
    display(out["summary"])
    print("20C component augmentation jitter:")
    display(out["temp20_jitter"])
    return out


def run_jacobian_sensitivity_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for base_name, base_spec in (
        ("R5_GATED", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_gate_smooth": 0.03,
        }),
        ("R5_GATED_BOUNDED", {
            "features": R5_BOUNDED_FEATURES,
            "kind": "bounded",
            "correction_limit": 0.05,
            "gated": True,
            "lambda_gate_smooth": 0.03,
        }),
    ):
        for lam in (1e-4, 5e-4, 1e-3):
            spec = dict(base_spec)
            spec.update({
                "lambda_jac": lam,
                "jacobian_batch_size": 128,
            })
            tag = f"{lam:.0e}".replace("-", "m")
            specs.append((f"{base_name}_JAC_{tag}", spec))
    out = _run_spec_grid(cfg, feature_frames, specs, prediction_path=cfg.output_dir / "jacobian_sensitivity_prediction_rows.csv")
    out["summary"].to_csv(cfg.output_dir / "jacobian_sensitivity_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "jacobian_sensitivity_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "jacobian_sensitivity_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_jacobian_jitter.csv", index=False)
    print("Jacobian sensitivity summary:")
    display(out["summary"])
    print("20C Jacobian jitter:")
    display(out["temp20_jitter"])
    return out


def run_seq_derivative_no_interaction_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for lam_seq in (0.2, 0.5):
        seq_tag = str(lam_seq).replace("0.", "p")
        specs.append((f"R5_SEQ_lseq_{seq_tag}", {
            "features": R5_FEATURES,
            "kind": "seq",
            "lambda_seq": lam_seq,
        }))
        specs.append((f"R5_GATED_SEQ_lseq_{seq_tag}", {
            "features": R5_GATED_FEATURES,
            "kind": "gated_seq",
            "lambda_seq": lam_seq,
            "lambda_gate_smooth": 0.03,
        }))
        for d1 in (0.05, 0.10):
            for d2 in (0.01, 0.03):
                tag = (
                    f"lseq_{seq_tag}"
                    f"_d1_{str(d1).replace('0.', 'p')}"
                    f"_d2_{str(d2).replace('0.', 'p')}"
                )
                specs.append((f"R5_GATED_SEQ_DERIV_{tag}", {
                    "features": R5_GATED_FEATURES,
                    "kind": "gated_seq",
                    "lambda_seq": lam_seq,
                    "lambda_d1": d1,
                    "lambda_d2": d2,
                    "lambda_gate_smooth": 0.03,
                }))
                specs.append((f"R5_GATED_SEQ_DERIV_BOUNDED_{tag}", {
                    "features": R5_BOUNDED_FEATURES,
                    "kind": "bounded",
                    "correction_limit": 0.05,
                    "gated": True,
                    "lambda_seq": lam_seq,
                    "lambda_d1": d1,
                    "lambda_d2": d2,
                    "lambda_gate_smooth": 0.03,
                }))
    out = _run_spec_grid(cfg, feature_frames, specs, prediction_path=cfg.output_dir / "seq_derivative_no_interaction_prediction_rows.csv")
    out["summary"].to_csv(cfg.output_dir / "seq_derivative_results_no_interaction.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "seq_derivative_by_temperature_no_interaction.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "seq_derivative_focus_metrics_no_interaction.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_seq_derivative_jitter.csv", index=False)
    print("Seq/derivative no-interaction summary:")
    display(out["summary"])
    print("20C seq/derivative jitter:")
    display(out["temp20_jitter"])
    return out


def run_internal_variance_reduction_experiments(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    outputs = {
        "bounded": run_bounded_correction_experiment(cfg),
        "component_aug": run_component_aug_consistency_experiment(cfg),
        "jacobian": run_jacobian_sensitivity_experiment(cfg),
        "seq_derivative": run_seq_derivative_no_interaction_experiment(cfg),
    }
    return outputs


class MultiScaleEndpointDataset(Dataset):
    def __init__(self, frames, feature_cols, scales=(50, 200, 500), stride=1, target_col="SOC_physical"):
        self.frames = []
        self.feature_cols = list(feature_cols)
        self.scales = tuple(int(s) for s in scales)
        self.max_scale = max(self.scales)
        self.stride = int(stride)
        self.index = []
        for fi, frame in enumerate(frames):
            frame = frame.reset_index(drop=True)
            cache = {
                "x": np.ascontiguousarray(frame[self.feature_cols].to_numpy(np.float32)),
                "y": np.ascontiguousarray(frame[target_col].to_numpy(np.float32)),
                "file_name": frame["file_name"].to_numpy(),
                "trajectory_id": frame["trajectory_id"].to_numpy(),
                "end_index": frame["end_index"].to_numpy(np.int64),
                "temperature": frame["temperature"].to_numpy(np.float32),
                "drive_cycle": frame["drive_cycle"].to_numpy(),
            }
            self.frames.append(cache)
            n = len(frame)
            if n < self.max_scale:
                continue
            for end in range(self.max_scale - 1, n, self.stride):
                if np.isfinite(cache["y"][end]):
                    self.index.append((fi, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, end = self.index[idx]
        f = self.frames[fi]
        xs = tuple(torch.from_numpy(f["x"][end - scale + 1:end + 1]) for scale in self.scales)
        y = torch.tensor([f["y"][end]], dtype=torch.float32)
        meta = {
            "file_name": f["file_name"][end],
            "trajectory_id": f["trajectory_id"][end],
            "end_index": int(f["end_index"][end]),
            "temperature": float(f["temperature"][end]),
            "drive_cycle": f["drive_cycle"][end],
        }
        return xs, y, meta


class EndpointBranchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        z = self.input_proj(x)
        out, _ = self.lstm(z)
        return self.norm(out + z)[:, -1, :]


class SharedEndpointConsistencySOC(nn.Module):
    def __init__(self, feature_cols, hidden_size=64, num_layers=1, dropout=0.0, gated=False):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.gated = bool(gated)
        self.encoder = EndpointBranchEncoder(
            len(self.feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, 1)
        self._init_gate()

    def _init_gate(self):
        if not self.gated:
            return
        self.component_names = ["V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        missing = [c for c in self.component_names if c not in self.feature_cols]
        if missing:
            raise ValueError(f"Gated endpoint model requires component columns: {missing}")
        gate_input_names = ["T", "V_raw", "V_corr_raw", "absI", "dI", "R0", "V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        self.gate_input_names = [c for c in gate_input_names if c in self.feature_cols]
        self.gate_input_indices = [self.feature_cols.index(c) for c in self.gate_input_names]
        self.component_indices = [self.feature_cols.index(c) for c in self.component_names]
        self.component_gate = nn.Sequential(
            nn.Linear(len(self.gate_input_indices), 32), nn.SiLU(),
            nn.Linear(32, 3), nn.Sigmoid()
        )

    def component_gates(self, x):
        if not self.gated:
            return None
        return self.component_gate(x[..., self.gate_input_indices])

    def apply_component_gates(self, x):
        if not self.gated:
            return x
        gates = self.component_gates(x)
        gated_by_index = {idx: j for j, idx in enumerate(self.component_indices)}
        cols = []
        for idx in range(x.size(-1)):
            col = x[..., idx:idx + 1]
            if idx in gated_by_index:
                col = col * gates[..., gated_by_index[idx]:gated_by_index[idx] + 1]
            cols.append(col)
        return torch.cat(cols, dim=-1)

    def forward_multi(self, xs):
        preds = []
        for x in xs:
            h = self.encoder(self.apply_component_gates(x))
            preds.append(torch.sigmoid(self.head(h)))
        branch_preds = torch.stack(preds, dim=1)
        return branch_preds.mean(dim=1), branch_preds


class MultiScaleLSTM_SOC(nn.Module):
    def __init__(self, feature_cols, scales=(50, 200, 500), hidden_size=64, num_layers=1, dropout=0.0, gated=False):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.scales = tuple(int(s) for s in scales)
        self.gated = bool(gated)
        self.branches = nn.ModuleList([
            EndpointBranchEncoder(
                len(self.feature_cols),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in self.scales
        ])
        self.branch_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in self.scales])
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size * len(self.scales), hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        self._init_gate()

    def _init_gate(self):
        if not self.gated:
            return
        self.component_names = ["V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        missing = [c for c in self.component_names if c not in self.feature_cols]
        if missing:
            raise ValueError(f"Gated multi-scale model requires component columns: {missing}")
        gate_input_names = ["T", "V_raw", "V_corr_raw", "absI", "dI", "R0", "V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        self.gate_input_names = [c for c in gate_input_names if c in self.feature_cols]
        self.gate_input_indices = [self.feature_cols.index(c) for c in self.gate_input_names]
        self.component_indices = [self.feature_cols.index(c) for c in self.component_names]
        self.component_gate = nn.Sequential(
            nn.Linear(len(self.gate_input_indices), 32), nn.SiLU(),
            nn.Linear(32, 3), nn.Sigmoid()
        )

    def component_gates(self, x):
        if not self.gated:
            return None
        return self.component_gate(x[..., self.gate_input_indices])

    def apply_component_gates(self, x):
        if not self.gated:
            return x
        gates = self.component_gates(x)
        gated_by_index = {idx: j for j, idx in enumerate(self.component_indices)}
        cols = []
        for idx in range(x.size(-1)):
            col = x[..., idx:idx + 1]
            if idx in gated_by_index:
                col = col * gates[..., gated_by_index[idx]:gated_by_index[idx] + 1]
            cols.append(col)
        return torch.cat(cols, dim=-1)

    def forward_multi(self, xs):
        hs = []
        branch_preds = []
        for x, branch, head in zip(xs, self.branches, self.branch_heads):
            h = branch(self.apply_component_gates(x))
            hs.append(h)
            branch_preds.append(torch.sigmoid(head(h)))
        fused = torch.sigmoid(self.fusion_head(torch.cat(hs, dim=-1)))
        return fused, torch.stack(branch_preds, dim=1)


def endpoint_consistency_loss(branch_preds):
    if branch_preds.size(1) < 2:
        return branch_preds.new_tensor(0.0)
    losses = []
    for i in range(branch_preds.size(1)):
        for j in range(i + 1, branch_preds.size(1)):
            losses.append(torch.mean(torch.abs(branch_preds[:, i, :] - branch_preds[:, j, :])))
    return torch.stack(losses).mean()


def make_multiscale_model(spec, cfg):
    common = {
        "feature_cols": spec["features"],
        "hidden_size": cfg.lstm_hidden_size,
        "num_layers": cfg.lstm_layers,
        "dropout": cfg.lstm_dropout,
        "gated": bool(spec.get("gated", False)),
    }
    if spec["kind"] == "endpoint_shared":
        return SharedEndpointConsistencySOC(**common)
    if spec["kind"] == "multiscale":
        return MultiScaleLSTM_SOC(scales=spec["scales"], **common)
    raise ValueError(f"Unknown multi-scale model kind: {spec['kind']}")


def multiscale_loss(model, xs, y, spec):
    pred, branch_preds = model.forward_multi(xs)
    if spec["kind"] == "endpoint_shared":
        y_exp = y[:, None, :].expand_as(branch_preds)
        l_soc = F.l1_loss(branch_preds, y_exp)
    else:
        l_soc = F.l1_loss(pred, y)
        if spec.get("branch_supervision", False):
            y_exp = y[:, None, :].expand_as(branch_preds)
            l_soc = l_soc + float(spec.get("lambda_branch", 0.5)) * F.l1_loss(branch_preds, y_exp)
    total = l_soc
    parts = {"soc": l_soc}
    if spec.get("lambda_endpoint", 0.0):
        l_endpoint = endpoint_consistency_loss(branch_preds)
        total = total + float(spec.get("lambda_endpoint", 0.0)) * l_endpoint
        parts["endpoint"] = l_endpoint
    if spec.get("lambda_aug", 0.0):
        xs_aug = [augment_components(x, spec["features"], spec) for x in xs]
        pred_aug, _ = model.forward_multi(xs_aug)
        l_aug = F.smooth_l1_loss(pred_aug, pred.detach(), beta=0.01)
        total = total + float(spec.get("lambda_aug", 0.0)) * l_aug
        parts["aug"] = l_aug
    parts["total"] = total
    return total, parts


@torch.no_grad()
def predict_multiscale_model(model, loader, cfg):
    model.eval()
    rows = []
    for xs, y, meta in loader:
        xs = [_move(x, cfg) for x in xs]
        pred, _ = model.forward_multi(xs)
        mdf = collate_meta_to_frame(meta)
        mdf["target_label"] = "physical"
        mdf["y_true"] = y.numpy()[:, 0]
        mdf["y_pred"] = pred.detach().cpu().numpy()[:, 0]
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def train_multiscale_model_from_spec(feature_frames, model_name, spec, cfg):
    feature_cols = spec["features"]
    scales = tuple(int(s) for s in spec["scales"])
    scaled, _ = _scale_frames(feature_frames, feature_cols)
    train_stride = int(getattr(cfg, "endpoint_train_stride", cfg.stride))
    train_ds = MultiScaleEndpointDataset(scaled["train"], feature_cols, scales, train_stride, target_col="SOC_physical")
    test_ds = MultiScaleEndpointDataset(scaled["test"], feature_cols, scales, cfg.stride, target_col="SOC_physical")
    if len(train_ds) == 0:
        raise ValueError(f"No train endpoints for multi-scale model {model_name}.")
    model = make_multiscale_model(spec, cfg).to(device)
    train_loader = _loader(train_ds, cfg, shuffle=True)
    test_loader = _loader(test_ds, cfg, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=cfg.lstm_weight_decay)
    history = []
    for ep in range(1, int(cfg.lstm_epochs) + 1):
        model.train()
        losses = []
        for xs, y, meta in train_loader:
            xs = [_move(x, cfg) for x in xs]
            y = _move(y, cfg)
            loss, _ = multiscale_loss(model, xs, y, spec)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": ep, "train_loss": float(np.mean(losses))}
        history.append(row)
        print_every = max(1, int(getattr(cfg, "lstm_print_every", 1)))
        if ep == 1 or ep == int(cfg.lstm_epochs) or ep % print_every == 0:
            print(f"{model_name} epoch={ep} train_loss={row['train_loss']:.5f}")
    pred_test = predict_multiscale_model(model, test_loader, cfg)
    return model, pd.DataFrame(history), pred_test


def _run_multiscale_spec_grid(cfg, feature_frames, specs, *, prediction_path=None, plot_dir=None):
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    max_scale = max(max(spec["scales"]) for _, spec in specs)
    pred_rows = []
    baseline_rows = _baseline_prediction_rows(cfg, min_endpoint=max_scale - 1)
    if len(baseline_rows):
        pred_rows.append(baseline_rows)
    histories = {}
    for model_name, spec in specs:
        print(f"\n=== endpoint/multiscale model: {model_name} ===")
        _, hist, pred_test = train_multiscale_model_from_spec(feature_frames, model_name, spec, cfg)
        histories[model_name] = hist
        pred_test = pred_test.assign(split="test", ablation=model_name)
        pred_rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=model_name, target_label="physical"))
    all_pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if prediction_path is not None:
        all_pred.to_csv(prediction_path, index=False)
    summary, by_temp, focus, temp20 = _summary_with_temp20_focus(all_pred)
    if plot_dir is not None:
        plot_temp20_multiscale_predictions(all_pred, cfg, plot_dir)
    return {
        "prediction_rows": all_pred,
        "summary": summary,
        "by_temperature": by_temp,
        "focus_metrics": focus,
        "temp20_jitter": temp20,
        "histories": histories,
    }


def plot_temp20_multiscale_predictions(pred_rows, cfg, out_dir_name="temp20_multiscale_prediction_plots"):
    out_dir = cfg.output_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pred_rows[np.isclose(pred_rows["temperature_C"].astype(float), 20.0)].copy()
    if df.empty:
        return
    for model, g in df.sort_values("end_index").groupby("model_name"):
        plt.figure(figsize=(10, 3.2))
        plt.plot(g["end_index"], g["y_true"] * 100.0, label="true SOC", linewidth=1.1)
        plt.plot(g["end_index"], g["y_pred"] * 100.0, label="predicted SOC", linewidth=0.9)
        plt.xlabel("time index")
        plt.ylabel("physical SOC (%SOC)")
        plt.title(f"FUDS 20C endpoint SOC | {model}")
        plt.legend()
        plt.tight_layout()
        safe = str(model).replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"{safe}_temp20_endpoint_prediction.png", dpi=180)
        plt.close()


def run_endpoint_consistency_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for lam in (0.05, 0.10, 0.20):
        tag = str(lam).replace("0.", "p")
        specs.append((f"R5_ENDPOINT_CONSIST_l{tag}", {
            "features": R5_FEATURES,
            "kind": "endpoint_shared",
            "scales": (50, 200, 500),
            "gated": False,
            "lambda_endpoint": lam,
        }))
        specs.append((f"R5_GATED_ENDPOINT_CONSIST_l{tag}", {
            "features": R5_GATED_FEATURES,
            "kind": "endpoint_shared",
            "scales": (50, 200, 500),
            "gated": True,
            "lambda_endpoint": lam,
        }))
    out = _run_multiscale_spec_grid(
        cfg,
        feature_frames,
        specs,
        prediction_path=cfg.output_dir / "endpoint_consistency_prediction_rows.csv",
        plot_dir="temp20_multiscale_prediction_plots",
    )
    out["summary"].to_csv(cfg.output_dir / "endpoint_consistency_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "endpoint_consistency_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "endpoint_consistency_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_endpoint_jitter_diagnostic.csv", index=False)
    print("Endpoint consistency summary:")
    display(out["summary"])
    return out


def run_multiscale_lstm_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    scale_sets = {
        "s50": (50,),
        "s50_200": (50, 200),
        "s50_200_500": (50, 200, 500),
    }
    specs = []
    for tag, scales in scale_sets.items():
        specs.append((f"R5_MULTISCALE_{tag}", {
            "features": R5_FEATURES,
            "kind": "multiscale",
            "scales": scales,
            "gated": False,
            "branch_supervision": False,
        }))
        specs.append((f"R5_GATED_MULTISCALE_{tag}", {
            "features": R5_GATED_FEATURES,
            "kind": "multiscale",
            "scales": scales,
            "gated": True,
            "branch_supervision": False,
        }))
    out = _run_multiscale_spec_grid(
        cfg,
        feature_frames,
        specs,
        prediction_path=cfg.output_dir / "multiscale_lstm_prediction_rows.csv",
        plot_dir="temp20_multiscale_prediction_plots",
    )
    out["summary"].to_csv(cfg.output_dir / "multiscale_lstm_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "multiscale_lstm_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "multiscale_lstm_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_multiscale_lstm_jitter.csv", index=False)
    print("Multi-scale LSTM summary:")
    display(out["summary"])
    return out


def run_multiscale_endpoint_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for lam in (0.05, 0.10, 0.20):
        tag = str(lam).replace("0.", "p")
        specs.append((f"R5_GATED_MULTISCALE_ENDPOINT_l{tag}", {
            "features": R5_GATED_FEATURES,
            "kind": "multiscale",
            "scales": (50, 200, 500),
            "gated": True,
            "branch_supervision": True,
            "lambda_branch": 0.5,
            "lambda_endpoint": lam,
        }))
    out = _run_multiscale_spec_grid(
        cfg,
        feature_frames,
        specs,
        prediction_path=cfg.output_dir / "multiscale_endpoint_prediction_rows.csv",
        plot_dir="temp20_multiscale_prediction_plots",
    )
    out["summary"].to_csv(cfg.output_dir / "multiscale_endpoint_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "multiscale_endpoint_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "multiscale_endpoint_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_multiscale_endpoint_jitter.csv", index=False)
    print("Multi-scale endpoint summary:")
    display(out["summary"])
    return out


def run_multiscale_endpoint_aug_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    feature_frames = _load_exp_c_feature_frames(cfg)
    specs = []
    for noise in (0.0025, 0.005):
        for drop in (0.02, 0.05):
            for lam_aug in (0.01, 0.03):
                name = (
                    "R5_GATED_MULTISCALE_ENDPOINT_AUG_WEAK"
                    f"_n{str(noise).replace('0.', 'p')}"
                    f"_d{str(drop).replace('0.', 'p')}"
                    f"_l{str(lam_aug).replace('0.', 'p')}"
                )
                specs.append((name, {
                    "features": R5_GATED_FEATURES,
                    "kind": "multiscale",
                    "scales": (50, 200, 500),
                    "gated": True,
                    "branch_supervision": True,
                    "lambda_branch": 0.5,
                    "lambda_endpoint": 0.10,
                    "lambda_aug": lam_aug,
                    "component_noise_std": noise,
                    "component_dropout_p": drop,
                }))
    out = _run_multiscale_spec_grid(
        cfg,
        feature_frames,
        specs,
        prediction_path=cfg.output_dir / "multiscale_endpoint_aug_prediction_rows.csv",
        plot_dir="temp20_multiscale_prediction_plots",
    )
    out["summary"].to_csv(cfg.output_dir / "multiscale_endpoint_aug_results.csv", index=False)
    out["by_temperature"].to_csv(cfg.output_dir / "multiscale_endpoint_aug_by_temperature.csv", index=False)
    out["focus_metrics"].to_csv(cfg.output_dir / "multiscale_endpoint_aug_focus_metrics.csv", index=False)
    out["temp20_jitter"].to_csv(cfg.output_dir / "temp20_multiscale_endpoint_aug_jitter.csv", index=False)
    print("Multi-scale endpoint weak augmentation summary:")
    display(out["summary"])
    return out


def run_endpoint_multiscale_experiments(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    outputs = {
        "endpoint": run_endpoint_consistency_experiment(cfg),
        "multiscale": run_multiscale_lstm_experiment(cfg),
        "multiscale_endpoint": run_multiscale_endpoint_experiment(cfg),
        "multiscale_endpoint_aug": run_multiscale_endpoint_aug_experiment(cfg),
    }
    comparison = pd.concat(
        [v["summary"].assign(experiment=k) for k, v in outputs.items()],
        ignore_index=True,
    )
    comparison.to_csv(cfg.output_dir / "endpoint_multiscale_comparison.csv", index=False)
    temp20 = pd.concat(
        [v["temp20_jitter"].assign(experiment=k) for k, v in outputs.items()],
        ignore_index=True,
    )
    temp20.to_csv(cfg.output_dir / "temp20_endpoint_jitter_diagnostic.csv", index=False)
    print("Endpoint/multi-scale comparison:")
    display(comparison.sort_values("temp20_MAE_pct").head(30))
    return outputs


def run_variance_control_experiment(cfg: CFG | None = None):
    cfg = make_cfg() if cfg is None else cfg
    configure_torch_runtime()
    cfg.decomposed_dir = cfg.output_dir / "decomposed_features_train_temp_minus10_0_10_25_50"
    feature_frames = load_feature_frame_dict_from_csv(cfg, decomposed_dir=cfg.decomposed_dir)
    feature_frames = add_temperature_rbf_features(feature_frames)
    feature_frames = add_interaction_features(feature_frames)

    baseline_path = cfg.output_dir / "train_temp_minus10_0_10_25_50_prediction_rows.csv"
    baseline_rows = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    if len(baseline_rows):
        baseline_rows = baseline_rows[baseline_rows["model_name"].isin(["R5_raw_I_T_all_components", "R5_GATED"])].copy()
        baseline_rows = baseline_rows[baseline_rows["label_type"] == "physical"].copy()

    model_names = [
        "R5_SEQ",
        "R5_GATED_SEQ",
        "R5_GATED_SEQ_DERIV",
        "R5_GATED_SEQ_DERIV_OVERLAP",
        "R5_GATED_BOUNDED",
        "R5_GATED_SEQ_BOUNDED",
        "R5_GATED_SEQ_BOUNDED_AUG",
        "R5_GATED_SEQ_R0_x_V_pol",
        "R5_GATED_SEQ_T_x_V_pol",
        "R5_GATED_SEQ_R0_x_absI",
        "R5_GATED_SEQ_V_pol_x_abs_dI",
    ]
    feature_lookup = build_prediction_feature_lookup(feature_frames)
    pred_rows = [baseline_rows] if len(baseline_rows) else []
    gate_rows = []
    histories = {}
    for name in model_names:
        print(f"\n=== variance-control model: {name} ===")
        _, hist, pred_test, gates = train_variance_model(feature_frames, name, cfg)
        histories[name] = hist
        pred_test = pred_test.assign(split="test", ablation=name)
        pred_rows.append(attach_prediction_features(pred_test, feature_lookup, ablation_name=name, target_label="physical"))
        if len(gates):
            gate_rows.append(gates)

    all_pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    all_pred.to_csv(cfg.output_dir / "variance_control_prediction_rows.csv", index=False)
    results = _overall_metrics(all_pred)
    by_temp = variance_by_temperature(all_pred)
    focus = focus_scope_metrics(all_pred, temperatures=(0.0, 10.0, 20.0))
    jitter_detail = _trajectory_jitter_rows(all_pred)
    temp20_jitter = by_temp[np.isclose(by_temp["temperature_C"].astype(float), 20.0)].copy()
    high_freq = jitter_detail[np.isclose(jitter_detail["temperature_C"].astype(float), 20.0)].copy()

    results.to_csv(cfg.output_dir / "variance_control_results.csv", index=False)
    by_temp.to_csv(cfg.output_dir / "variance_control_by_temperature.csv", index=False)
    focus.to_csv(cfg.output_dir / "variance_control_focus_metrics.csv", index=False)
    temp20_jitter.to_csv(cfg.output_dir / "temp20_jitter_diagnostic.csv", index=False)
    high_freq.to_csv(cfg.output_dir / "temp20_high_frequency_error.csv", index=False)
    if gate_rows:
        gates = pd.concat(gate_rows, ignore_index=True)
        gates.to_csv(cfg.output_dir / "variance_control_gate_by_temperature.csv", index=False)
        plot_component_gate_summary(gates, cfg)
        gate_plot = cfg.output_dir / "component_gate_by_temperature.png"
        if gate_plot.exists():
            gate_plot.replace(cfg.output_dir / "variance_control_gate_by_temperature.png")
    plot_temp20_predictions(all_pred, cfg)

    print("Variance-control overall results:")
    display(results)
    print("20C jitter diagnostic:")
    display(temp20_jitter)
    print("0/10/20C focus metrics:")
    display(focus)
    return {
        "prediction_rows": all_pred,
        "results": results,
        "by_temperature": by_temp,
        "focus_metrics": focus,
        "temp20_jitter": temp20_jitter,
        "high_frequency_error": high_freq,
        "histories": histories,
    }
