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
from .deep_no_leak_experiment import CausalConvBlock, SequenceWindowDataset, make_eval_loader
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


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_remote_screen_seed0"


@dataclass
class GoalScreenConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 180
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 64
    lambda_rex: float = 2.0
    rex_group: str = "temperature_drive"
    huber_beta: float = 0.02
    sequence_loss: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 20
    valid_every: int = 10
    variant_set: str = "stage1"
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class Variant:
    name: str
    recurrent: str = "lstm"
    layers: int = 2
    head_kind: str = "mlp"
    temp_mode: str = "none"
    dropout: float = 0.05
    lambda_rex: float = 2.0
    lr: float = 8e-4
    weight_0: float = 1.0
    weight_25: float = 1.0
    weight_45: float = 1.0
    kernel_size: int = 5
    norm_kind: str = "channel"
    lambda_bias: float = 0.0


class ResidualMLPHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, residual_scale: float):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.residual_scale = float(residual_scale)
        self.residual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual_scale * torch.tanh(self.residual(x))


class GatedLinearMLPHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, gate_bias: float | None = None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size // 2, 1),
        )
        if gate_bias is not None:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, float(gate_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        return (1.0 - gate) * self.linear(x) + gate * self.mlp(x)


class VcorrITGoalModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        recurrent: str = "lstm",
        layers: int = 2,
        head_kind: str = "mlp",
        temp_mode: str = "none",
        dropout: float = 0.05,
        kernel_size: int = 5,
        norm_kind: str = "channel",
    ):
        super().__init__()
        self.temp_idx = 2
        self.head_kind = str(head_kind)
        self.temp_mode = str(temp_mode)
        self.recurrent = str(recurrent).lower()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        if self.recurrent == "tcn":
            self.tcn_blocks = nn.Sequential(
                *[
                    CausalConvBlock(
                        hidden_size,
                        kernel_size=int(kernel_size),
                        dilation=2 ** i,
                        dropout=float(dropout),
                        norm_kind=str(norm_kind),
                    )
                    for i in range(int(layers))
                ]
            )
        elif self.recurrent == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=4 * hidden_size,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_pos = nn.Parameter(torch.zeros(1, 512, hidden_size))
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        elif self.recurrent in {"lstm", "gru", "rnn"}:
            if self.recurrent == "lstm":
                self.rnn = nn.LSTM(
                    hidden_size,
                    hidden_size,
                    num_layers=int(layers),
                    batch_first=True,
                    dropout=float(dropout) if int(layers) > 1 else 0.0,
                )
            elif self.recurrent == "gru":
                self.rnn = nn.GRU(
                    hidden_size,
                    hidden_size,
                    num_layers=int(layers),
                    batch_first=True,
                    dropout=float(dropout) if int(layers) > 1 else 0.0,
                )
            else:
                self.rnn = nn.RNN(
                    hidden_size,
                    hidden_size,
                    num_layers=int(layers),
                    nonlinearity="tanh",
                    batch_first=True,
                    dropout=float(dropout) if int(layers) > 1 else 0.0,
                )
        else:
            raise ValueError(f"Unknown recurrent={recurrent}")
        self.norm = nn.LayerNorm(hidden_size)
        self.base_head = self._make_head(hidden_size, head_kind, dropout)
        if self.temp_mode == "bias":
            self.temp_bias = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.temp_bias[-1].weight)
            nn.init.zeros_(self.temp_bias[-1].bias)
        elif self.temp_mode == "hard_affine":
            self.temp_affine_log_scale = nn.Parameter(torch.zeros(3))
            self.temp_affine_bias = nn.Parameter(torch.zeros(3))
        elif self.temp_mode == "moe":
            self.expert_heads = nn.ModuleList([self._make_head(hidden_size, head_kind, dropout) for _ in range(3)])
            self.temp_gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 3))
        elif self.temp_mode == "hard_heads":
            self.expert_heads = nn.ModuleList([self._make_head(hidden_size, head_kind, dropout) for _ in range(3)])
        elif self.temp_mode != "none":
            raise ValueError(f"Unknown temp_mode={temp_mode}")

    @staticmethod
    def _make_head(hidden_size: int, head_kind: str, dropout: float) -> nn.Module:
        if head_kind == "linear":
            return nn.Linear(hidden_size, 1)
        if head_kind == "mlp":
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_size, 1),
            )
        if head_kind == "residual_mlp_0p5":
            return ResidualMLPHead(hidden_size, dropout, residual_scale=0.5)
        if head_kind == "residual_mlp_1p0":
            return ResidualMLPHead(hidden_size, dropout, residual_scale=1.0)
        if head_kind == "residual_mlp_2p0":
            return ResidualMLPHead(hidden_size, dropout, residual_scale=2.0)
        if head_kind == "gated_mlp":
            return GatedLinearMLPHead(hidden_size, dropout)
        if head_kind == "gated_mlp_linstart":
            return GatedLinearMLPHead(hidden_size, dropout, gate_bias=-2.0)
        if head_kind == "gated_mlp_midstart":
            return GatedLinearMLPHead(hidden_size, dropout, gate_bias=-1.0)
        raise ValueError(f"Unknown head_kind={head_kind}")

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        if self.recurrent == "tcn":
            out = self.tcn_blocks(z.transpose(1, 2)).transpose(1, 2)
            return self.norm(out)
        if self.recurrent == "transformer":
            if z.shape[1] > self.transformer_pos.shape[1]:
                raise RuntimeError(f"Transformer window length {z.shape[1]} exceeds positional limit {self.transformer_pos.shape[1]}.")
            out = self.transformer(z + self.transformer_pos[:, : z.shape[1], :])
            return self.norm(out)
        out, _ = self.rnn(z)
        return self.norm(out)

    def _hard_head_indices(self, temp_scaled: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(temp_scaled[..., 0], dtype=torch.long)
        idx = torch.where(temp_scaled[..., 0] > -0.45, torch.ones_like(idx), idx)
        idx = torch.where(temp_scaled[..., 0] > 0.65, torch.full_like(idx, 2), idx)
        return idx

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        temp = x[..., self.temp_idx:self.temp_idx + 1]
        if self.temp_mode == "none":
            return self.base_head(h)
        if self.temp_mode == "bias":
            return self.base_head(h) + self.temp_bias(temp)
        if self.temp_mode == "hard_affine":
            base = self.base_head(h)
            idx = self._hard_head_indices(temp)
            flat_idx = idx.reshape(-1)
            scale = torch.exp(self.temp_affine_log_scale.clamp(-1.0, 1.0)).to(base.device, base.dtype)
            bias = self.temp_affine_bias.to(base.device, base.dtype)
            selected_scale = scale.index_select(0, flat_idx).reshape_as(base)
            selected_bias = bias.index_select(0, flat_idx).reshape_as(base)
            return selected_scale * base + selected_bias
        if self.temp_mode == "moe":
            logits = torch.stack([head(h) for head in self.expert_heads], dim=-1).squeeze(-2)
            gate = torch.softmax(self.temp_gate(temp), dim=-1)
            return torch.sum(logits * gate, dim=-1, keepdim=True)
        if self.temp_mode == "hard_heads":
            logits = torch.stack([head(h) for head in self.expert_heads], dim=-1).squeeze(-2)
            idx = self._hard_head_indices(temp)
            return torch.gather(logits, dim=-1, index=idx.unsqueeze(-1))
        raise AssertionError("unreachable")

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def variant_list(variant_set: str = "stage1") -> list[Variant]:
    if variant_set == "stage2":
        return [
            Variant(
                "tempmix_linear_lstm2_rex2_w0x2_w25x2",
                recurrent="lstm",
                layers=2,
                head_kind="linear",
                temp_mode="moe",
                lambda_rex=2.0,
                weight_0=2.0,
                weight_25=2.0,
            ),
            Variant(
                "tempmix_linear_lstm2_rex2_w0x2_w25x3",
                recurrent="lstm",
                layers=2,
                head_kind="linear",
                temp_mode="moe",
                lambda_rex=2.0,
                weight_0=2.0,
                weight_25=3.0,
            ),
            Variant(
                "tempmix_linear_lstm2_rex0p5_w0x2_w25x2",
                recurrent="lstm",
                layers=2,
                head_kind="linear",
                temp_mode="moe",
                lambda_rex=0.5,
                weight_0=2.0,
                weight_25=2.0,
            ),
            Variant(
                "tempmix_mlp_tcn4_rex2",
                recurrent="tcn",
                layers=4,
                head_kind="mlp",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=2.0,
            ),
            Variant(
                "tempmix_linear_tcn4_rex2",
                recurrent="tcn",
                layers=4,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=2.0,
            ),
            Variant(
                "tempmix_linear_tcn5_rex2_w0x2_w25x2",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=2.0,
                weight_0=2.0,
                weight_25=2.0,
            ),
            Variant(
                "tempbias_mlp_tcn5_rex2",
                recurrent="tcn",
                layers=5,
                head_kind="mlp",
                temp_mode="bias",
                dropout=0.04,
                lambda_rex=2.0,
            ),
            Variant(
                "baseline_mlp_tcn5_rex2",
                recurrent="tcn",
                layers=5,
                head_kind="mlp",
                temp_mode="none",
                dropout=0.04,
                lambda_rex=2.0,
            ),
        ]
    if variant_set == "stage3":
        return [
            Variant(
                "tempbias_mlp_tcn5_rex0p5",
                recurrent="tcn",
                layers=5,
                head_kind="mlp",
                temp_mode="bias",
                dropout=0.04,
                lambda_rex=0.5,
            ),
            Variant(
                "tempbias_mlp_tcn5_rex0_w25x2",
                recurrent="tcn",
                layers=5,
                head_kind="mlp",
                temp_mode="bias",
                dropout=0.04,
                lambda_rex=0.0,
                weight_25=2.0,
            ),
            Variant(
                "tempmix_linear_tcn5_rex0p5_w25x3",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=0.5,
                weight_25=3.0,
            ),
            Variant(
                "tempmix_linear_tcn5_rex0_w25x3",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=0.0,
                weight_25=3.0,
            ),
            Variant(
                "hardheads_mlp_tcn5_rex2",
                recurrent="tcn",
                layers=5,
                head_kind="mlp",
                temp_mode="hard_heads",
                dropout=0.04,
                lambda_rex=2.0,
            ),
            Variant(
                "hardheads_mlp_tcn5_rex0p5_w25x2",
                recurrent="tcn",
                layers=5,
                head_kind="mlp",
                temp_mode="hard_heads",
                dropout=0.04,
                lambda_rex=0.5,
                weight_25=2.0,
            ),
            Variant(
                "hardheads_linear_tcn5_rex2",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="hard_heads",
                dropout=0.04,
                lambda_rex=2.0,
            ),
            Variant(
                "hardheads_linear_tcn5_rex0p5_w25x2",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="hard_heads",
                dropout=0.04,
                lambda_rex=0.5,
                weight_25=2.0,
            ),
            Variant(
                "tempbias_mlp_tcn4_k3_rex2",
                recurrent="tcn",
                layers=4,
                head_kind="mlp",
                temp_mode="bias",
                dropout=0.04,
                lambda_rex=2.0,
                kernel_size=3,
            ),
            Variant(
                "tempmix_linear_tcn4_k3_rex2_w25x2",
                recurrent="tcn",
                layers=4,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=2.0,
                weight_25=2.0,
                kernel_size=3,
            ),
        ]
    if variant_set == "stage4":
        return [
            Variant(
                "tempmix_linear_lstm2_rex2_biasrex25",
                recurrent="lstm",
                layers=2,
                head_kind="linear",
                temp_mode="moe",
                lambda_rex=2.0,
                lambda_bias=25.0,
            ),
            Variant(
                "tempmix_linear_lstm2_rex2_biasrex50",
                recurrent="lstm",
                layers=2,
                head_kind="linear",
                temp_mode="moe",
                lambda_rex=2.0,
                lambda_bias=50.0,
            ),
            Variant(
                "tempmix_linear_lstm2_rex0p5_biasrex25",
                recurrent="lstm",
                layers=2,
                head_kind="linear",
                temp_mode="moe",
                lambda_rex=0.5,
                lambda_bias=25.0,
            ),
            Variant(
                "tempbias_mlp_lstm2_rex2_biasrex25",
                recurrent="lstm",
                layers=2,
                head_kind="mlp",
                temp_mode="bias",
                lambda_rex=2.0,
                lambda_bias=25.0,
            ),
            Variant(
                "tempmix_linear_tcn5_rex2_biasrex25",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=2.0,
                lambda_bias=25.0,
            ),
            Variant(
                "tempmix_linear_tcn5_rex2_biasrex50",
                recurrent="tcn",
                layers=5,
                head_kind="linear",
                temp_mode="moe",
                dropout=0.04,
                lambda_rex=2.0,
                lambda_bias=50.0,
            ),
        ]
    return [
        Variant("baseline_mlp_lstm2_rex2", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="none", lambda_rex=2.0),
        Variant("tempbias_mlp_lstm2_rex2", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="bias", lambda_rex=2.0),
        Variant("tempmix_mlp_lstm2_rex2", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="moe", lambda_rex=2.0),
        Variant("hardheads_mlp_lstm2_rex2", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="hard_heads", lambda_rex=2.0),
        Variant("tempmix_linear_lstm2_rex2", recurrent="lstm", layers=2, head_kind="linear", temp_mode="moe", lambda_rex=2.0),
        Variant("tempmix_mlp_lstm2_rex0p5", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="moe", lambda_rex=0.5),
        Variant("tempmix_mlp_lstm2_w25x2", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="moe", lambda_rex=2.0, weight_25=2.0),
        Variant("tempmix_mlp_lstm2_w25x3", recurrent="lstm", layers=2, head_kind="mlp", temp_mode="moe", lambda_rex=2.0, weight_25=3.0),
        Variant("tempmix_mlp_gru2_rex2", recurrent="gru", layers=2, head_kind="mlp", temp_mode="moe", lambda_rex=2.0),
        Variant("hardheads_mlp_gru2_w25x2", recurrent="gru", layers=2, head_kind="mlp", temp_mode="hard_heads", lambda_rex=2.0, weight_25=2.0),
    ]


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


def evaluate_loader_mae(model: nn.Module, loader) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y, _meta in loader:
            pred = model(x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
            yy = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            errors.append(torch.abs(pred - yy).detach().cpu().numpy())
    return float(np.mean(np.concatenate(errors))) if errors else float("nan")


def temp_weights(meta, variant: Variant, batch_size: int) -> torch.Tensor:
    vals = meta.get("temperature")
    if torch.is_tensor(vals):
        temps = vals.to(device=device, dtype=torch.float32).reshape(-1)
        if int(temps.numel()) == int(batch_size):
            weights = torch.ones_like(temps)
            weights = torch.where(torch.isclose(temps, torch.zeros_like(temps), atol=1e-6), torch.full_like(weights, float(variant.weight_0)), weights)
            weights = torch.where(torch.isclose(temps, torch.full_like(temps, 25.0), atol=1e-6), torch.full_like(weights, float(variant.weight_25)), weights)
            weights = torch.where(torch.isclose(temps, torch.full_like(temps, 45.0), atol=1e-6), torch.full_like(weights, float(variant.weight_45)), weights)
            return weights
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


def group_keys(meta, group_name: str) -> list[str] | torch.Tensor:
    group_tensor_key = {
        "temperature": "group_temperature",
        "drive": "group_drive",
        "temperature_drive": "group_temperature_drive",
    }.get(str(group_name))
    if group_tensor_key and group_tensor_key in meta and torch.is_tensor(meta[group_tensor_key]):
        return meta[group_tensor_key].to(device=device, dtype=torch.long).reshape(-1)
    temps = [float(v) for v in meta["temperature"]]
    drives = [str(v) for v in meta["drive_cycle"]]
    if group_name == "temperature":
        return [f"T{t:g}" for t in temps]
    if group_name == "drive":
        return [f"D{d}" for d in drives]
    return [f"T{t:g}_{d}" for t, d in zip(temps, drives)]


def train_variant(
    cfg: GoalScreenConfig,
    variant: Variant,
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
    train_cls = SequenceWindowDataset if cfg.sequence_loss else DecomposedWindowDataset
    train_ds = train_cls(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    model = VcorrITGoalModel(
        input_dim=len(FEATURE_COLS),
        hidden_size=int(cfg.hidden_size),
        recurrent=variant.recurrent,
        layers=variant.layers,
        head_kind=variant.head_kind,
        temp_mode=variant.temp_mode,
        dropout=variant.dropout,
        kernel_size=variant.kernel_size,
        norm_kind=variant.norm_kind,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    best_valid_mae = float("inf")
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
            if cfg.sequence_loss:
                pred = model.forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
                sample_signed_error = (pred - y).mean(dim=(1, 2))
            else:
                pred = model(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
                sample_signed_error = (pred - y).mean(dim=1)
            sw = temp_weights(meta, variant, int(sample_loss.numel()))
            keys = group_keys(meta, cfg.rex_group)
            group_losses = []
            group_weights = []
            group_biases = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                g_bias = sample_signed_error.index_select(0, idx).mean()
                g_weight = sw.index_select(0, idx).mean()
                group_losses.append(g_loss)
                group_biases.append(g_bias)
                group_weights.append(g_weight)
                by_group.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(group_losses) if group_losses else sample_loss.mean().view(1)
            bias_stack = torch.stack(group_biases) if group_biases else sample_signed_error.mean().view(1)
            wstack = torch.stack(group_weights) if group_weights else stack.new_ones(stack.shape)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            bias_var = bias_stack.var(unbiased=False) if len(bias_stack) > 1 else bias_stack.new_tensor(0.0)
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_bias) * bias_var
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
            "bias_var": float(bias_var.detach().cpu()),
        }
        for key, vals in by_group.items():
            safe = key.replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.valid_every)) == 0:
            valid_mae = evaluate_loader_mae(model, valid_loader)
            row["valid_MAE"] = float(valid_mae)
            row["valid_MAE_pct"] = float(valid_mae * 100.0)
            if np.isfinite(valid_mae) and valid_mae < best_valid_mae:
                best_valid_mae = float(valid_mae)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            valid_msg = f" valid_mae={row['valid_MAE_pct']:.3f}%" if "valid_MAE_pct" in row else ""
            print(f"{model_name} epoch={ep} loss={row['loss']:.5f}{valid_msg}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        best_epoch = int(cfg.epochs)
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"{model_name}_history.csv", index=False)
    lookup = make_endpoint_lookup(frames, FEATURE_COLS)
    pred = attach_eval_features(predict_loader(model, test_loader, model_name), lookup)
    valid_pred = attach_eval_features(predict_loader(model, valid_loader, model_name), lookup)
    for df in (pred, valid_pred):
        df["seed"] = int(cfg.seed)
        df["variant"] = variant.name
        df["selected_epoch"] = int(best_epoch)
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
            df["input_feature_dim"] = len(FEATURE_COLS)
            df["goal_0C_lt1_25C_lt0p7"] = bool(
                not by_temp.empty
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 0.0), "MAE_pct"].min() < 1.0)
                and (by_temp.loc[np.isclose(by_temp["temperature_C"], 25.0), "MAE_pct"].min() < 0.7)
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


def write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, by_temp, valid_overall):
    pass0 = by_temp[np.isclose(by_temp["temperature_C"], 0.0)].copy()
    pass25 = by_temp[np.isclose(by_temp["temperature_C"], 25.0)].copy()
    lines = [
        "# NMC Vcorr/I/T Goal Screening",
        "",
        "## Goal",
        "- Inputs fixed to V_corr_raw, I_raw, T.",
        "- seq_len=50, hidden/model dimension=64.",
        "- Target: FUDS 0C MAE < 1.0%p and 25C MAE < 0.7%p.",
        "",
        "## Overall",
        table_md(overall, ["variant", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## By temperature",
        table_md(by_temp, ["variant", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio", "selected_epoch", "goal_0C_lt1_25C_lt0p7"]),
        "",
        "## Valid overall",
        table_md(valid_overall, ["variant", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Leakage audit",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## Input schema",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## R0 / Vcorr preprocessing",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: GoalScreenConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("This experiment is fixed to hidden_size=64 and window_len=50.")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_screen_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    set_seed(cfg.seed)
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    start_audit = write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
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
        "variants": [asdict(v) for v in variants],
        "variant_set": str(cfg.variant_set),
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, by_temp, valid_overall)
    print("Goal screen overall:")
    print(overall.to_string(index=False), flush=True)
    print("Goal screen by temperature:")
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
    p = argparse.ArgumentParser(description="NMC Vcorr/I/T fixed-input goal screening.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=GoalScreenConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--valid-every", type=int, default=10)
    p.add_argument("--variant-set", default="stage1", choices=["stage1", "stage2", "stage3", "stage4"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GoalScreenConfig(
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
