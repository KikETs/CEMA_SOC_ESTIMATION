from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import make_cfg
from .deep_no_leak_experiment import make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset, collate_meta_to_frame
from .nmc_branchbands_experiment import (
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
    focus_metrics,
    metrics_by_trajectory,
    table_md,
    write_start_audit,
)
from .nmc_vcorr_it_lstm_singlehead_bytemp import (
    FEATURE_COLS,
    attach_eval_features,
    make_endpoint_lookup,
)
from .nmc_vit_feature_lstm_experiment import (
    add_vit_engineered_features,
    write_input_schema,
    write_leakage_audit,
)
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_remote_designed_seed0"


@dataclass
class DesignedConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 120
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 20
    valid_every: int = 5
    variant_set: str = "designed"
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class DesignedVariant:
    name: str
    encoder: str = "lstm"
    layers: int = 1
    temp_mode: str = "bias"
    use_anchor: bool = True
    use_mha: bool = True
    correction_limit: float = 0.8
    dropout: float = 0.04
    lambda_rex: float = 2.0
    lr: float = 8e-4
    weight_0: float = 1.0
    weight_25: float = 1.0
    weight_45: float = 1.0
    selection: str = "goal"
    lambda_anchor: float = 0.0
    lambda_mono: float = 0.0
    lambda_delta: float = 0.0
    anchor_low_i_A: float = 0.20
    anchor_low_di_A: float = 0.40


DESIGN_INTENT = [
    {
        "component": "static Vcorr/T anchor",
        "why": (
            "FUDS contains low-excitation and plateau-like regions where the 50 s current response is weak; "
            "the model therefore needs a voltage/temperature SOC anchor that does not depend on integrating current."
        ),
    },
    {
        "component": "unidirectional RNN encoder",
        "why": (
            "V_corr alone is ambiguous under load history, so the encoder reads the causal transient response to "
            "instantaneous current over the last 50 s. It never sees current beyond the endpoint."
        ),
    },
    {
        "component": "last-query multi-head attention",
        "why": (
            "Only parts of a window are informative. Attention lets the endpoint representation emphasize relaxation "
            "or current-change segments instead of averaging all 50 s equally."
        ),
    },
    {
        "component": "bounded dynamic correction",
        "why": (
            "The dynamic branch should correct the static SOC estimate when excitation is visible, but should not "
            "freely override the voltage anchor in poorly observable windows."
        ),
    },
    {
        "component": "temperature-conditioned bias/MoE",
        "why": (
            "The OCV/dynamic response map changes with temperature, while temperature is known online. The gate uses "
            "T and window-local excitation summaries only, not SOC labels or cumulative charge."
        ),
    },
    {
        "component": "endpoint-only training",
        "why": (
            "Full-window attention is valid for endpoint online estimation because the complete 50 s history is known. "
            "Endpoint-only loss avoids using later window samples to supervise earlier intermediate timestamps."
        ),
    },
]


class AuxWindowDataset(torch.utils.data.Dataset):
    def __init__(self, frames, feature_cols, window_len, stride, target_label="physical"):
        self.frames = []
        self.feature_cols = list(feature_cols)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.target_label = str(target_label)
        self.index = []
        for fi, frame in enumerate(frames):
            frame = frame.reset_index(drop=True)
            abs_i = frame["absI"].to_numpy(np.float32) if "absI" in frame.columns else np.abs(frame["I_raw"].to_numpy(np.float32))
            d_i = frame["dI"].to_numpy(np.float32) if "dI" in frame.columns else np.diff(frame["I_raw"].to_numpy(np.float32), prepend=frame["I_raw"].iloc[0])
            cache = {
                "x": np.ascontiguousarray(frame[self.feature_cols].to_numpy(np.float32)),
                "y_physical": np.ascontiguousarray(frame["SOC_physical"].to_numpy(np.float32)),
                "file_name": frame["file_name"].to_numpy(),
                "trajectory_id": frame["trajectory_id"].to_numpy(),
                "end_index": frame["end_index"].to_numpy(np.int64),
                "temperature": frame["temperature"].to_numpy(np.float32),
                "drive_cycle": frame["drive_cycle"].to_numpy(),
                "absI": np.ascontiguousarray(abs_i),
                "absdI": np.ascontiguousarray(np.abs(d_i).astype(np.float32)),
                "V_corr_raw_unscaled": np.ascontiguousarray(frame["V_corr_raw"].to_numpy(np.float32)),
            }
            self.frames.append(cache)
            n = len(frame)
            if n < self.window_len:
                continue
            for start in range(0, n - self.window_len + 1, self.stride):
                end = start + self.window_len - 1
                if np.isfinite(cache["y_physical"][end]):
                    self.index.append((fi, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        y_val = float(f["y_physical"][end])
        if y_val < 0.2:
            soc_bin = 0
        elif y_val <= 0.8:
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
            "mean_absI": float(np.mean(f["absI"][start:end + 1])),
            "end_absI": float(f["absI"][end]),
            "mean_absdI": float(np.mean(f["absdI"][start:end + 1])),
            "vcorr_span_raw": float(np.max(f["V_corr_raw_unscaled"][start:end + 1]) - np.min(f["V_corr_raw_unscaled"][start:end + 1])),
        }
        return (
            torch.from_numpy(f["x"][start:end + 1]),
            torch.as_tensor([y_val], dtype=torch.float32),
            meta,
        )


class LastQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        query = h[:, -1:, :]
        ctx, _weights = self.attn(query, h, h, need_weights=False)
        return self.norm(ctx[:, 0, :] + h[:, -1, :])


class DesignedRNNModel(nn.Module):
    def __init__(self, variant: DesignedVariant, input_dim: int = 3, hidden_size: int = 64):
        super().__init__()
        self.variant = variant
        self.hidden_size = int(hidden_size)
        self.temp_idx = 2
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        rnn_cls = nn.GRU if variant.encoder == "gru" else nn.LSTM
        self.encoder = rnn_cls(
            hidden_size,
            hidden_size,
            num_layers=int(variant.layers),
            batch_first=True,
            dropout=float(variant.dropout) if int(variant.layers) > 1 else 0.0,
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.pool = LastQueryAttention(hidden_size, variant.dropout) if variant.use_mha else None
        self.static_anchor = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(variant.dropout)),
            nn.Linear(hidden_size, 1),
        )
        dyn_in = hidden_size + 6
        self.dynamic_head = nn.Sequential(
            nn.Linear(dyn_in, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(variant.dropout)),
            nn.Linear(hidden_size, 1),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(dyn_in, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(variant.dropout)),
            nn.Linear(hidden_size, 1),
        )
        gate_in = 7
        self.correction_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        if variant.temp_mode == "bias":
            self.temp_bias = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.temp_bias[-1].weight)
            nn.init.zeros_(self.temp_bias[-1].bias)
        elif variant.temp_mode == "moe":
            self.expert_delta = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dyn_in, hidden_size),
                        nn.SiLU(),
                        nn.Dropout(float(variant.dropout)),
                        nn.Linear(hidden_size, 1),
                    )
                    for _ in range(3)
                ]
            )
            self.temp_gate = nn.Sequential(nn.Linear(gate_in, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 3))
        elif variant.temp_mode == "hard":
            self.expert_delta = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dyn_in, hidden_size),
                        nn.SiLU(),
                        nn.Dropout(float(variant.dropout)),
                        nn.Linear(hidden_size, 1),
                    )
                    for _ in range(3)
                ]
            )
        elif variant.temp_mode != "none":
            raise ValueError(f"Unknown temp_mode={variant.temp_mode}")

    def window_stats(self, x: torch.Tensor) -> torch.Tensor:
        v = x[..., 0]
        i = x[..., 1]
        t = x[..., 2]
        di = i[:, 1:] - i[:, :-1]
        abs_i_mean = i.abs().mean(dim=1, keepdim=True)
        i_std = i.std(dim=1, keepdim=True, unbiased=False)
        abs_di_mean = di.abs().mean(dim=1, keepdim=True)
        v_span = (v.max(dim=1).values - v.min(dim=1).values).unsqueeze(1)
        v_delta = (v[:, -1] - v[:, 0]).unsqueeze(1)
        t_last = t[:, -1:].contiguous()
        return torch.cat([abs_i_mean, i_std, abs_di_mean, v_span, v_delta, t_last], dim=1)

    def _hard_idx(self, t_last: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(t_last[:, 0], dtype=torch.long)
        idx = torch.where(t_last[:, 0] > -0.45, torch.ones_like(idx), idx)
        idx = torch.where(t_last[:, 0] > 0.65, torch.full_like(idx, 2), idx)
        return idx

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.input_proj(x)
        h, _state = self.encoder(z)
        h = self.encoder_norm(h)
        pooled = self.pool(h) if self.pool is not None else h[:, -1, :]
        stats = self.window_stats(x)
        dyn = torch.cat([pooled, stats], dim=1)
        gate_features = torch.cat([stats, x[:, -1:, 0]], dim=1)
        corr_gate = torch.sigmoid(self.correction_gate(gate_features))

        if self.variant.temp_mode == "moe":
            deltas = torch.cat([head(dyn) for head in self.expert_delta], dim=1)
            weights = torch.softmax(self.temp_gate(gate_features), dim=1)
            delta = torch.sum(deltas * weights, dim=1, keepdim=True)
        elif self.variant.temp_mode == "hard":
            deltas = torch.cat([head(dyn) for head in self.expert_delta], dim=1)
            idx = self._hard_idx(stats[:, 5:6])
            delta = torch.gather(deltas, dim=1, index=idx.unsqueeze(1))
        else:
            delta = self.dynamic_head(dyn)

        anchor_in = torch.cat([x[:, -1:, 0], x[:, -1:, self.temp_idx]], dim=1)
        anchor_logit = self.static_anchor(anchor_in)
        correction = corr_gate * float(self.variant.correction_limit) * torch.tanh(delta)
        if self.variant.use_anchor:
            logit = anchor_logit + correction
        else:
            logit = self.direct_head(dyn)
        if self.variant.temp_mode == "bias":
            logit = logit + self.temp_bias(stats[:, 5:6])
        return torch.sigmoid(logit), torch.sigmoid(anchor_logit), correction, corr_gate

    def monotonic_penalty(self, x: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
        anchor_in = torch.cat([x[:, -1:, 0], x[:, -1:, self.temp_idx]], dim=1)
        anchor_hi = anchor_in.clone()
        anchor_hi[:, 0:1] = anchor_hi[:, 0:1] + float(eps)
        lo = self.static_anchor(anchor_in)
        hi = self.static_anchor(anchor_hi)
        return F.relu(lo - hi).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(x)[0]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def variant_list(variant_set: str = "designed") -> list[DesignedVariant]:
    if variant_set == "tri45_valid":
        return [
            DesignedVariant(
                "tri45_lstm_mha_anchor_tempmoe_lim0p8_w25x2_w45x3",
                encoder="lstm",
                layers=1,
                temp_mode="moe",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.8,
                lambda_rex=2.0,
                weight_25=2.0,
                weight_45=3.0,
                lambda_anchor=1.0,
                lambda_mono=0.5,
                lambda_delta=1e-3,
                selection="tri45_goal",
            ),
            DesignedVariant(
                "tri45_lstm_mha_anchor_tempmoe_lim1p0_w25x2_w45x4",
                encoder="lstm",
                layers=1,
                temp_mode="moe",
                use_anchor=True,
                use_mha=True,
                correction_limit=1.0,
                lambda_rex=2.0,
                weight_25=2.0,
                weight_45=4.0,
                lambda_anchor=1.0,
                lambda_mono=0.5,
                lambda_delta=1e-3,
                selection="tri45_goal",
            ),
            DesignedVariant(
                "tri45_gru_mha_anchor_tempmoe_lim0p8_w25x2_w45x3",
                encoder="gru",
                layers=1,
                temp_mode="moe",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.8,
                lambda_rex=2.0,
                weight_25=2.0,
                weight_45=3.0,
                lambda_anchor=1.0,
                lambda_mono=0.5,
                lambda_delta=1e-3,
                selection="tri45_goal",
            ),
            DesignedVariant(
                "tri45_lstm_mha_anchor_hardheads_lim0p8_w25x2_w45x3",
                encoder="lstm",
                layers=1,
                temp_mode="hard",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.8,
                lambda_rex=2.0,
                weight_25=2.0,
                weight_45=3.0,
                lambda_anchor=1.0,
                lambda_mono=0.5,
                lambda_delta=1e-3,
                selection="tri45_goal",
            ),
            DesignedVariant(
                "tri45_lstm_mha_anchor_tempbias_lim0p8_w25x2_w45x3",
                encoder="lstm",
                layers=1,
                temp_mode="bias",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.8,
                lambda_rex=2.0,
                weight_25=2.0,
                weight_45=3.0,
                lambda_anchor=1.0,
                lambda_mono=0.5,
                lambda_delta=1e-3,
                selection="tri45_goal",
            ),
        ]
    if variant_set == "anchor_loss":
        return [
            DesignedVariant(
                "anchorloss_lstm_tempbias_lim0p4_a2_mono1_d1e3",
                encoder="lstm",
                layers=1,
                temp_mode="bias",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.4,
                lambda_rex=2.0,
                lambda_anchor=2.0,
                lambda_mono=1.0,
                lambda_delta=1e-3,
                selection="goal",
            ),
            DesignedVariant(
                "anchorloss_lstm_tempbias_lim0p6_a2_mono1_d1e3",
                encoder="lstm",
                layers=1,
                temp_mode="bias",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.6,
                lambda_rex=2.0,
                lambda_anchor=2.0,
                lambda_mono=1.0,
                lambda_delta=1e-3,
                selection="goal",
            ),
            DesignedVariant(
                "anchorloss_lstm_tempbias_lim0p8_a4_mono2_d2e3",
                encoder="lstm",
                layers=1,
                temp_mode="bias",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.8,
                lambda_rex=2.0,
                lambda_anchor=4.0,
                lambda_mono=2.0,
                lambda_delta=2e-3,
                selection="goal",
            ),
            DesignedVariant(
                "anchorloss_lstm_tempmoe_lim0p6_a2_mono1_d1e3",
                encoder="lstm",
                layers=1,
                temp_mode="moe",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.6,
                lambda_rex=2.0,
                lambda_anchor=2.0,
                lambda_mono=1.0,
                lambda_delta=1e-3,
                selection="goal",
            ),
            DesignedVariant(
                "anchorloss_lstm_tempbias_lim0p4_a1_mono0_d5e4",
                encoder="lstm",
                layers=1,
                temp_mode="bias",
                use_anchor=True,
                use_mha=True,
                correction_limit=0.4,
                lambda_rex=2.0,
                lambda_anchor=1.0,
                lambda_mono=0.0,
                lambda_delta=5e-4,
                selection="goal",
            ),
        ]
    return [
        DesignedVariant(
            "lstm_mha_anchor_tempbias_lim0p8_goal",
            encoder="lstm",
            layers=1,
            temp_mode="bias",
            use_anchor=True,
            use_mha=True,
            correction_limit=0.8,
            lambda_rex=2.0,
            selection="goal",
        ),
        DesignedVariant(
            "gru_mha_anchor_tempbias_lim0p8_goal",
            encoder="gru",
            layers=1,
            temp_mode="bias",
            use_anchor=True,
            use_mha=True,
            correction_limit=0.8,
            lambda_rex=2.0,
            selection="goal",
        ),
        DesignedVariant(
            "lstm_mha_anchor_tempmoe_lim1p0_goal",
            encoder="lstm",
            layers=1,
            temp_mode="moe",
            use_anchor=True,
            use_mha=True,
            correction_limit=1.0,
            lambda_rex=2.0,
            selection="goal",
        ),
        DesignedVariant(
            "gru_mha_anchor_tempmoe_lim1p0_goal",
            encoder="gru",
            layers=1,
            temp_mode="moe",
            use_anchor=True,
            use_mha=True,
            correction_limit=1.0,
            lambda_rex=2.0,
            selection="goal",
        ),
        DesignedVariant(
            "lstm_mha_anchor_hardheads_lim1p0_w25x1p5_goal",
            encoder="lstm",
            layers=1,
            temp_mode="hard",
            use_anchor=True,
            use_mha=True,
            correction_limit=1.0,
            lambda_rex=2.0,
            weight_25=1.5,
            selection="goal",
        ),
        DesignedVariant(
            "lstm_mha_noanchor_tempmoe_goal",
            encoder="lstm",
            layers=1,
            temp_mode="moe",
            use_anchor=False,
            use_mha=True,
            correction_limit=1.0,
            lambda_rex=2.0,
            selection="goal",
        ),
    ]


def temp_weights(meta, variant: DesignedVariant, batch_size: int) -> torch.Tensor:
    weights = []
    for t in [float(v) for v in meta["temperature"]]:
        if abs(t - 0.0) < 1e-6:
            weights.append(float(variant.weight_0))
        elif abs(t - 25.0) < 1e-6:
            weights.append(float(variant.weight_25))
        elif abs(t - 45.0) < 1e-6:
            weights.append(float(variant.weight_45))
        else:
            weights.append(1.0)
    if len(weights) != batch_size:
        weights = [1.0] * batch_size
    return torch.as_tensor(weights, device=device, dtype=torch.float32)


def group_keys(meta, group_name: str) -> list[str]:
    temps = [float(v) for v in meta["temperature"]]
    drives = [str(v) for v in meta["drive_cycle"]]
    if group_name == "temperature":
        return [f"T{t:g}" for t in temps]
    if group_name == "drive":
        return [f"D{d}" for d in drives]
    return [f"T{t:g}_{d}" for t, d in zip(temps, drives)]


def predict_loader(model: nn.Module, loader, model_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, meta in loader:
            pred = model(x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
            mdf = collate_meta_to_frame(meta)
            mdf["model_name"] = model_name
            mdf["target_label"] = "physical"
            mdf["y_true"] = y.numpy()[:, 0]
            mdf["y_pred"] = pred.detach().cpu().numpy()[:, 0]
            rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
    return out


def valid_score(overall: pd.DataFrame, by_temp: pd.DataFrame, variant: DesignedVariant) -> float:
    if overall.empty or by_temp.empty:
        return float("inf")
    overall_mae = float(overall["MAE_pct"].iloc[0])
    def t_mae(temp: float) -> float:
        sub = by_temp[np.isclose(by_temp["temperature_C"], temp)]
        return float(sub["MAE_pct"].iloc[0]) if len(sub) else overall_mae
    mae0 = t_mae(0.0)
    mae25 = t_mae(25.0)
    mae45 = t_mae(45.0)
    if variant.selection == "overall":
        return overall_mae
    if variant.selection == "tri45_goal":
        target0 = 1.0
        target25 = 0.7
        target45 = 0.3
        excess = (
            max(0.0, mae0 - target0)
            + 1.5 * max(0.0, mae25 - target25)
            + 2.0 * max(0.0, mae45 - target45)
        )
        normalized_worst = max(mae0 / target0, mae25 / target25, mae45 / target45)
        return float(normalized_worst + 3.0 * excess + 0.05 * overall_mae)
    return float(mae25 + 0.7 * mae0 + 0.2 * overall_mae + 2.0 * max(0.0, mae0 - 1.0))


def evaluate_valid(model: nn.Module, loader, model_name: str, lookup) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = attach_eval_features(predict_loader(model, loader, model_name), lookup)
    return pred, _overall_metrics(pred), variance_by_temperature(pred)


def anchor_weights(meta, variant: DesignedVariant, batch_size: int) -> torch.Tensor:
    if "mean_absI" not in meta:
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    mean_abs_i = torch.as_tensor(meta["mean_absI"], device=device, dtype=torch.float32)
    mean_abs_di = torch.as_tensor(meta["mean_absdI"], device=device, dtype=torch.float32)
    wi = torch.exp(-mean_abs_i / max(float(variant.anchor_low_i_A), 1e-6))
    wdi = torch.exp(-mean_abs_di / max(float(variant.anchor_low_di_A), 1e-6))
    return (0.25 + wi * wdi).clamp(min=0.05, max=1.25)


def train_variant(
    cfg: DesignedConfig,
    variant: DesignedVariant,
    frames: dict[str, list[pd.DataFrame]],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    model_name = f"{cfg.output_prefix}_{variant.name}"
    scaled, _ = make_scaled_frames_for_ablation(frames, FEATURE_COLS)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    train_ds = AuxWindowDataset(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = AuxWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = AuxWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    model = DesignedRNNModel(variant, input_dim=len(FEATURE_COLS), hidden_size=int(cfg.hidden_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    lookup = make_endpoint_lookup(frames, FEATURE_COLS)
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        by_group: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred, anchor_pred, correction, _corr_gate = model.forward_parts(x)
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            sw = temp_weights(meta, variant, int(sample_loss.numel()))
            keys = group_keys(meta, cfg.rex_group)
            group_losses = []
            group_weights = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                g_weight = sw.index_select(0, idx).mean()
                group_losses.append(g_loss)
                group_weights.append(g_weight)
                by_group.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(group_losses) if group_losses else sample_loss.mean().view(1)
            wstack = torch.stack(group_weights) if group_weights else stack.new_ones(stack.shape)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(variant.lambda_rex) * rex_var
            if float(variant.lambda_anchor) > 0.0:
                aw = anchor_weights(meta, variant, int(sample_loss.numel()))
                anchor_sample = F.smooth_l1_loss(anchor_pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
                anchor_loss = (anchor_sample * aw).sum() / aw.sum().clamp_min(1e-6)
                loss = loss + float(variant.lambda_anchor) * anchor_loss
            else:
                anchor_loss = loss.new_tensor(0.0)
            if float(variant.lambda_mono) > 0.0:
                mono_loss = model.monotonic_penalty(x)
                loss = loss + float(variant.lambda_mono) * mono_loss
            else:
                mono_loss = loss.new_tensor(0.0)
            if float(variant.lambda_delta) > 0.0:
                delta_loss = correction.square().mean()
                loss = loss + float(variant.lambda_delta) * delta_loss
            else:
                delta_loss = loss.new_tensor(0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {
            "model_name": model_name,
            "variant": variant.name,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "mono_loss": float(mono_loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
        }
        for key, vals in by_group.items():
            safe = key.replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.valid_every)) == 0:
            valid_pred, valid_overall, valid_by_temp = evaluate_valid(model, valid_loader, model_name, lookup)
            score = valid_score(valid_overall, valid_by_temp, variant)
            row["valid_score"] = float(score)
            row["valid_MAE_pct"] = float(valid_overall["MAE_pct"].iloc[0]) if len(valid_overall) else float("nan")
            for temp in (0.0, 25.0, 45.0):
                sub = valid_by_temp[np.isclose(valid_by_temp["temperature_C"], temp)]
                row[f"valid_T{int(temp)}_MAE_pct"] = float(sub["MAE_pct"].iloc[0]) if len(sub) else float("nan")
            if np.isfinite(score) and score < best_score:
                best_score = float(score)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            valid_msg = f" valid_score={row['valid_score']:.3f} valid_mae={row['valid_MAE_pct']:.3f}%" if "valid_score" in row else ""
            print(f"{model_name} epoch={ep} loss={row['loss']:.5f}{valid_msg}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        best_epoch = int(cfg.epochs)
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"{model_name}_history.csv", index=False)
    pred = attach_eval_features(predict_loader(model, test_loader, model_name), lookup)
    valid_pred = attach_eval_features(predict_loader(model, valid_loader, model_name), lookup)
    for df in (pred, valid_pred):
        df["seed"] = int(cfg.seed)
        df["variant"] = variant.name
        df["selected_epoch"] = int(best_epoch)
        df["selection"] = variant.selection
        df["input_feature_dim"] = len(FEATURE_COLS)
    pred.to_csv(out_dir / f"{model_name}_prediction_rows.csv.gz", index=False, compression="gzip")
    valid_pred.to_csv(out_dir / f"{model_name}_valid_prediction_rows.csv.gz", index=False, compression="gzip")
    overall = _overall_metrics(pred)
    by_temp = variance_by_temperature(pred)
    by_traj = metrics_by_trajectory(pred)
    focus = focus_metrics(pred, cfg, model_name)
    valid_overall = _overall_metrics(valid_pred)
    valid_by_temp = variance_by_temperature(valid_pred)
    for df in (overall, by_temp, by_traj, focus, valid_overall, valid_by_temp):
        if not df.empty:
            df["variant"] = variant.name
            df["selected_epoch"] = int(best_epoch)
            df["selection"] = variant.selection
            df["input_feature_dim"] = len(FEATURE_COLS)
            df["goal_0C_lt1_25C_lt0p7"] = bool(
                not by_temp.empty
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 0.0), "MAE_pct"].min() < 1.0)
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 25.0), "MAE_pct"].min() < 0.7)
            )
            df["goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = bool(
                not by_temp.empty
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 0.0), "MAE_pct"].min() < 1.0)
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 25.0), "MAE_pct"].min() < 0.7)
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 45.0), "MAE_pct"].min() < 0.3)
            )
    overall.to_csv(out_dir / f"{model_name}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{model_name}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{model_name}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{model_name}_focus.csv", index=False)
    valid_overall.to_csv(out_dir / f"{model_name}_valid_overall.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{model_name}_valid_by_temperature.csv", index=False)
    print("Overall:")
    print(overall.to_string(index=False), flush=True)
    print("By temperature:")
    print(by_temp.to_string(index=False), flush=True)
    return {
        "history": history_df,
        "pred": pred,
        "valid_pred": valid_pred,
        "overall": overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
    }


def write_report(cfg, out_dir, r0_df, schema, leakage, overall, by_temp, valid_overall):
    intent_df = pd.DataFrame(DESIGN_INTENT)
    lines = [
        "# NMC Vcorr/I/T Designed RNN-MHA Goal Experiment",
        "",
        "## Fixed Goal Conditions",
        "- Inputs: V_corr_raw, I_raw, T only.",
        "- Window length: 50.",
        "- Model dimension: 64.",
        "- Test: train DST/US06, valid VALIDATION, test FUDS.",
        "- Target: 0C MAE < 1.0%p, 25C MAE < 0.7%p, and 45C MAE < 0.3%p.",
        "- Checkpoint selection: validation only; FUDS test is evaluated after selection.",
        "- Excluded: switching independently selected checkpoints by temperature.",
        "",
        "## Design Intent",
        table_md(intent_df, ["component", "why"]),
        "",
        "## Overall",
        table_md(overall, ["variant", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch", "selection"]),
        "",
        "## By Temperature",
        table_md(by_temp, ["variant", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio", "selected_epoch", "selection", "goal_0C_lt1_25C_lt0p7_45C_lt0p3"]),
        "",
        "## Valid Overall",
        table_md(valid_overall, ["variant", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch", "selection"]),
        "",
        "## Leakage Audit",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## Input Schema",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## R0 / Vcorr Preprocessing",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: DesignedConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("This experiment is fixed to hidden_size=64 and window_len=50.")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_designed_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    set_seed(cfg.seed)
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    schema = write_input_schema(FEATURE_COLS, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    leakage = write_leakage_audit(FEATURE_COLS, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    results = []
    variants = variant_list(cfg.variant_set)
    for variant in variants:
        set_seed(cfg.seed)
        print(f"===== variant {variant.name} =====", flush=True)
        results.append(train_variant(cfg, variant, frames, out_dir))
    overall = pd.concat([r["overall"] for r in results], ignore_index=True).sort_values("MAE_pct")
    by_temp = pd.concat([r["by_temperature"] for r in results], ignore_index=True)
    focus = pd.concat([r["focus"] for r in results], ignore_index=True)
    valid_overall = pd.concat([r["valid_overall"] for r in results], ignore_index=True).sort_values("MAE_pct")
    valid_by_temp = pd.concat([r["valid_by_temperature"] for r in results], ignore_index=True)
    history = pd.concat([r["history"] for r in results], ignore_index=True)
    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    valid_overall.to_csv(out_dir / f"{cfg.output_prefix}_valid_overall.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{cfg.output_prefix}_valid_by_temperature.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "hidden_size_verified": 64,
        "window_len_verified": 50,
        "sequence_loss": False,
        "endpoint_only_attention": True,
        "design_intent": DESIGN_INTENT,
        "variant_set": str(cfg.variant_set),
        "variants": [asdict(v) for v in variants],
        "checkpoint_selection": "validation_only",
        "temperature_checkpoint_switching": False,
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, r0_df, schema, leakage, overall, by_temp, valid_overall)
    print("Designed experiment overall:")
    print(overall.to_string(index=False), flush=True)
    print("Designed experiment by temperature:")
    print(by_temp.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    return {
        "overall": overall,
        "by_temperature": by_temp,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC fixed Vcorr/I/T designed RNN-MHA experiment.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=DesignedConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--valid-every", type=int, default=5)
    p.add_argument("--variant-set", default="designed", choices=["designed", "anchor_loss", "tri45_valid"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DesignedConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        valid_every=int(args.valid_every),
        variant_set=str(args.variant_set),
    )
    run(cfg)


if __name__ == "__main__":
    main()
