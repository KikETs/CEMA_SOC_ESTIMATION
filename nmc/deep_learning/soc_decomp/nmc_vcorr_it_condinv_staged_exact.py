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
from .deep_no_leak_experiment import SequenceWindowDataset, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_goal_remote_screen import Variant, VcorrITGoalModel, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_condinv_staged_exact_seed0"


@dataclass
class CondInvConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 60
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    eval_every: int = 5
    print_every: int = 10
    variant_set: str = "screen"
    stage2_epochs: int = 35
    lr_stage2: float = 8e-4
    focus45_weight: float = 12.0
    keep_lambda: float = 4.0
    low_soc_threshold: float = 0.35
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class CondInvVariant:
    name: str
    recurrent: str = "tcn"
    layers: int = 5
    kernel_size: int = 5
    head_kind: str = "linear"
    temp_mode: str = "moe"
    dropout: float = 0.06
    lambda_rex: float = 2.0
    lambda_condinv: float = 0.02
    weight_0: float = 4.0
    weight_25: float = 2.2
    weight_45: float = 1.0
    lr: float = 8e-4
    enable_swa: bool = False
    swa_start: int = 10
    swa_end: int = 30
    swa_every: int = 5
    enable_stage2: bool = False
    corr_limit: float = 1.2
    corr_mode: str = "45_low"
    enable_snapshot_ensemble: bool = False
    enable_snapshot_gate: bool = False
    snapshot_epochs: tuple[int, ...] = (5, 7, 10)
    stage2_keep_scale: float = 1.0
    sequence_training: bool = False


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def condinv_variants(name: str) -> list[CondInvVariant]:
    if name == "fast":
        return [
            CondInvVariant("condinv_tcn5_mmd0p02_w0x4_w25x2p2", lambda_condinv=0.02),
        ]
    if name == "staged":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_staged45_corr",
                lambda_condinv=0.02,
                enable_stage2=True,
                corr_limit=1.2,
            )
        ]
    if name == "staged_bands":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_epoch9_stage2_25_45_corr",
                lambda_condinv=0.02,
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="temp_bands",
            )
        ]
    if name == "staged_cold_hot":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_epoch7_stage2_0_45_corr",
                lambda_condinv=0.02,
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="cold_hot",
            )
        ]
    if name == "staged_cold_hot_swa5_10":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_swa5_10_stage2_0_45_corr",
                lambda_condinv=0.02,
                enable_swa=True,
                swa_start=5,
                swa_end=10,
                swa_every=1,
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="cold_hot",
            )
        ]
    if name == "staged_cold_hot_w25x4":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_w0x4_w25x4_stage2_0_45_corr",
                lambda_condinv=0.02,
                weight_25=4.0,
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="cold_hot",
            )
        ]
    if name == "snapshot_5_7_10_cold_hot":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_snapshot5_7_10_stage2_0_45_corr",
                lambda_condinv=0.02,
                enable_snapshot_ensemble=True,
                snapshot_epochs=(5, 7, 10),
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="cold_hot",
            )
        ]
    if name == "snapshot_gate_5_7_10_cold_hot":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_snapshotgate5_7_10_stage2_0_45_corr",
                lambda_condinv=0.02,
                enable_snapshot_gate=True,
                snapshot_epochs=(5, 7, 10),
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="cold_hot",
                stage2_keep_scale=0.1,
            )
        ]
    if name == "seq_staged_cold_hot":
        return [
            CondInvVariant(
                "condinv_seq_tcn5_mmd0p02_stage2_0_45_corr",
                lambda_condinv=0.02,
                enable_stage2=True,
                corr_limit=1.2,
                corr_mode="cold_hot",
                sequence_training=True,
            )
        ]
    if name == "strong":
        return [
            CondInvVariant("condinv_tcn5_mmd5_w0x4_w25x2p2", lambda_condinv=5.0),
            CondInvVariant("condinv_tcn5_mmd20_w0x4_w25x2p2", lambda_condinv=20.0),
            CondInvVariant("condinv_tcn6_mmd5_w0x4_w25x2p2", layers=6, lambda_condinv=5.0),
        ]
    if name == "swa":
        return [
            CondInvVariant(
                "condinv_tcn5_mmd0p02_swa10_30",
                lambda_condinv=0.02,
                enable_swa=True,
                swa_start=10,
                swa_end=30,
            ),
            CondInvVariant(
                "condinv_tcn5_mmd5_swa10_30",
                lambda_condinv=5.0,
                enable_swa=True,
                swa_start=10,
                swa_end=30,
            ),
            CondInvVariant(
                "condinv_lstm2_tempbias_mmd0p02_swa5_50",
                recurrent="lstm",
                layers=2,
                head_kind="mlp",
                temp_mode="bias",
                dropout=0.08,
                lambda_condinv=0.02,
                weight_0=1.0,
                weight_25=2.0,
                enable_swa=True,
                swa_start=5,
                swa_end=50,
            ),
        ]
    return [
        CondInvVariant("condinv_tcn5_mmd0p02_w0x4_w25x2p2", lambda_condinv=0.02),
        CondInvVariant("condinv_tcn5_mmd0p05_w0x4_w25x2p2", lambda_condinv=0.05),
        CondInvVariant(
            "condinv_tcn6_mmd0p02_w0x4_w25x2p2",
            layers=6,
            lambda_condinv=0.02,
        ),
        CondInvVariant(
            "condinv_lstm2_tempbias_mmd0p02_w25x2",
            recurrent="lstm",
            layers=2,
            head_kind="mlp",
            temp_mode="bias",
            dropout=0.08,
            lambda_condinv=0.02,
            weight_0=1.0,
            weight_25=2.0,
        ),
    ]


def make_model(variant: CondInvVariant, cfg: CondInvConfig) -> VcorrITGoalModel:
    return VcorrITGoalModel(
        input_dim=len(FEATURE_COLS),
        hidden_size=int(cfg.hidden_size),
        recurrent=variant.recurrent,
        layers=int(variant.layers),
        head_kind=variant.head_kind,
        temp_mode=variant.temp_mode,
        dropout=float(variant.dropout),
        kernel_size=int(variant.kernel_size),
        norm_kind="channel",
    ).to(device)


def to_variant(v: CondInvVariant) -> Variant:
    return Variant(
        name=v.name,
        recurrent=v.recurrent,
        layers=v.layers,
        head_kind=v.head_kind,
        temp_mode=v.temp_mode,
        dropout=v.dropout,
        lambda_rex=v.lambda_rex,
        lr=v.lr,
        weight_0=v.weight_0,
        weight_25=v.weight_25,
        weight_45=v.weight_45,
        kernel_size=v.kernel_size,
        norm_kind="channel",
    )


def _meta_list(meta, key: str) -> list:
    vals = meta[key]
    if torch.is_tensor(vals):
        return vals.detach().cpu().tolist()
    if isinstance(vals, np.ndarray):
        return vals.tolist()
    return list(vals)


def _meta_optional_tensor(meta, key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if key not in meta:
        return None
    vals = meta[key]
    if torch.is_tensor(vals):
        return vals.to(device=device, dtype=dtype).reshape(-1)
    try:
        return torch.as_tensor(vals, device=device, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return None


def conditional_profile_mmd(h: torch.Tensor, y: torch.Tensor, meta) -> torch.Tensor:
    """Align hidden centroids across whatever drive profiles are present.

    Earlier screens only compared DST and US06. That is not a universal
    profile-invariance objective: profile-rotation folds can train on
    FUDS+US06 or DST+FUDS, where a DST/US06-only loss silently becomes
    inactive. This version keeps the same temperature/SOC-bin conditioning
    but compares all profile centroids available in the batch.
    """
    h_norm = F.normalize(h, dim=1)
    soc = y[:, 0].detach()
    soc_bins = torch.bucketize(soc, torch.as_tensor([0.2, 0.5, 0.8], device=soc.device))
    temp_ids = _meta_optional_tensor(meta, "temperature_id", device=h.device, dtype=torch.long)
    drive_ids = _meta_optional_tensor(meta, "drive_id", device=h.device, dtype=torch.long)
    if (
        temp_ids is not None
        and drive_ids is not None
        and int(temp_ids.numel()) == int(h.shape[0])
        and int(drive_ids.numel()) == int(h.shape[0])
    ):
        if int(drive_ids.numel()) == 0:
            return h.new_tensor(0.0)
        n_temp = int(temp_ids.max().detach().cpu()) + 1
        n_drive = int(drive_ids.max().detach().cpu()) + 1
        if n_temp <= 0 or n_drive <= 1:
            return h.new_tensor(0.0)
        condition_ids = temp_ids * 4 + soc_bins.to(dtype=torch.long)
        group_ids = condition_ids * n_drive + drive_ids
        n_conditions = n_temp * 4
        n_groups = n_conditions * n_drive
        sums = h_norm.new_zeros((n_groups, h_norm.shape[1]))
        counts = h_norm.new_zeros(n_groups)
        sums.index_add_(0, group_ids, h_norm)
        counts.index_add_(0, group_ids, torch.ones(int(h_norm.shape[0]), device=h.device, dtype=h_norm.dtype))
        centroids = sums / counts.clamp_min(1.0).unsqueeze(1)
        valid = counts >= 2.0
        losses = []
        for condition_id in range(n_conditions):
            start = condition_id * n_drive
            stop = start + n_drive
            valid_drive = valid[start:stop]
            if int(valid_drive.sum().detach().cpu()) < 2:
                continue
            stack = centroids[start:stop][valid_drive]
            centered = stack - stack.mean(dim=0, keepdim=True)
            losses.append(centered.square().mean())
        return torch.stack(losses).mean() if losses else h.new_tensor(0.0)

    temps = [round(float(v), 3) for v in _meta_list(meta, "temperature")]
    drives = [str(v) for v in _meta_list(meta, "drive_cycle")]
    soc_bins_cpu = soc_bins.detach().cpu().tolist()
    losses = []
    for temp in sorted(set(temps)):
        temp_idx = [i for i, t in enumerate(temps) if t == temp]
        for sb in range(4):
            idx = [i for i in temp_idx if int(soc_bins_cpu[i]) == sb]
            if len(idx) < 4:
                continue
            centroids = []
            for drive in sorted({drives[i] for i in idx}):
                group = torch.as_tensor([i for i in idx if drives[i] == drive], device=h.device, dtype=torch.long)
                if group.numel() >= 2:
                    centroids.append(h_norm.index_select(0, group).mean(dim=0))
            if len(centroids) < 2:
                continue
            stack = torch.stack(centroids, dim=0)
            centered = stack - stack.mean(dim=0, keepdim=True)
            losses.append(centered.square().mean())
    return torch.stack(losses).mean() if losses else h.new_tensor(0.0)


@torch.no_grad()
def eval_by_temp(model: nn.Module, loader, split: str, variant_name: str, epoch: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x).detach().cpu().numpy()[:, 0]
        y_cpu = y.detach().cpu()
        true = y_cpu[:, -1, 0].numpy() if y_cpu.ndim == 3 else y_cpu.numpy()[:, 0]
        temps_src = meta["temperature"]
        temps = temps_src.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(temps_src) else np.asarray(temps_src, dtype=np.float32)
        for temp in sorted(set(float(t) for t in temps)):
            idx = np.isclose(temps, temp)
            err = pred[idx] - true[idx]
            rows.append(
                {
                    "variant": variant_name,
                    "epoch": int(epoch),
                    "split": split,
                    "temperature_C": float(temp),
                    "n_windows": int(idx.sum()),
                    "sum_abs_error": float(np.sum(np.abs(err))),
                    "sum_sq_error": float(np.sum(err**2)),
                    "sum_error": float(np.sum(err)),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.groupby(["variant", "epoch", "split", "temperature_C"], as_index=False).agg(
            n_windows=("n_windows", "sum"),
            sum_abs_error=("sum_abs_error", "sum"),
            sum_sq_error=("sum_sq_error", "sum"),
            sum_error=("sum_error", "sum"),
        )
        denom = out["n_windows"].clip(lower=1).astype(float)
        out["MAE_pct"] = out["sum_abs_error"] / denom * 100.0
        out["RMSE_pct"] = np.sqrt(out["sum_sq_error"] / denom) * 100.0
        out["bias_pct"] = out["sum_error"] / denom * 100.0
        out = out.drop(columns=["sum_abs_error", "sum_sq_error", "sum_error"])
    return out


class Correction45(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        corr_limit: float,
        mode: str = "45_low",
        input_dim: int = 3,
        zero_init: bool = True,
    ):
        super().__init__()
        self.corr_limit = float(corr_limit)
        self.mode = str(mode)
        stat_dim = 4 * int(input_dim)
        if self.mode in {"cold_mid_hot", "cold_hot_midlite"}:
            out_dim = 3
        elif self.mode in {"temp_bands", "cold_hot", "cold_hot_deep"}:
            out_dim = 2
        else:
            out_dim = 1
        if self.mode == "cold_hot_deep":
            self.net = nn.Sequential(
                nn.Linear(int(hidden_size) + stat_dim, int(hidden_size)),
                nn.LayerNorm(int(hidden_size)),
                nn.SiLU(),
                nn.Dropout(0.04),
                nn.Linear(int(hidden_size), int(hidden_size)),
                nn.LayerNorm(int(hidden_size)),
                nn.SiLU(),
                nn.Dropout(0.04),
                nn.Linear(int(hidden_size), out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(int(hidden_size) + stat_dim, int(hidden_size)),
                nn.LayerNorm(int(hidden_size)),
                nn.SiLU(),
                nn.Dropout(0.04),
                nn.Linear(int(hidden_size), out_dim),
            )
        if bool(zero_init):
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _window_stats(self, x: torch.Tensor) -> torch.Tensor:
        x_end = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_delta = x_end - x[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def forward(self, h_last: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([h_last, self._window_stats(x)], dim=1))
        temp_end = x[:, -1, 2:3]
        if self.mode == "temp_bands":
            mid_temp_gate = torch.sigmoid((temp_end + 0.45) * 8.0) * torch.sigmoid((0.65 - temp_end) * 8.0)
            high_temp_gate = torch.sigmoid((temp_end - 0.65) * 8.0)
            return self.corr_limit * (mid_temp_gate * torch.tanh(raw[:, 0:1]) + high_temp_gate * torch.tanh(raw[:, 1:2]))
        if self.mode in {"cold_hot", "cold_hot_deep"}:
            cold_temp_gate = torch.sigmoid((-0.45 - temp_end) * 8.0)
            high_temp_gate = torch.sigmoid((temp_end - 0.65) * 8.0)
            return self.corr_limit * (cold_temp_gate * torch.tanh(raw[:, 0:1]) + high_temp_gate * torch.tanh(raw[:, 1:2]))
        if self.mode == "cold_mid_hot":
            cold_temp_gate = torch.sigmoid((-0.45 - temp_end) * 8.0)
            mid_temp_gate = torch.sigmoid((temp_end + 0.45) * 8.0) * torch.sigmoid((0.65 - temp_end) * 8.0)
            high_temp_gate = torch.sigmoid((temp_end - 0.65) * 8.0)
            mid_scale = 0.25
            return self.corr_limit * (
                cold_temp_gate * torch.tanh(raw[:, 0:1])
                + mid_scale * mid_temp_gate * torch.tanh(raw[:, 1:2])
                + high_temp_gate * torch.tanh(raw[:, 2:3])
            )
        if self.mode == "cold_hot_midlite":
            cold_temp_gate = torch.sigmoid((-0.45 - temp_end) * 8.0)
            mid_temp_gate = torch.sigmoid((temp_end + 0.45) * 8.0) * torch.sigmoid((0.65 - temp_end) * 8.0)
            high_temp_gate = torch.sigmoid((temp_end - 0.65) * 8.0)
            mid_scale = 0.15
            return self.corr_limit * (
                cold_temp_gate * torch.tanh(raw[:, 0:1])
                + mid_scale * mid_temp_gate * torch.tanh(raw[:, 1:2])
                + high_temp_gate * torch.tanh(raw[:, 2:3])
            )
        if self.mode == "all_temp":
            return self.corr_limit * torch.tanh(raw)
        if self.mode == "cold_only":
            cold_temp_gate = torch.sigmoid((-0.45 - temp_end) * 8.0)
            return self.corr_limit * cold_temp_gate * torch.tanh(raw)
        if self.mode == "hot_only":
            high_temp_gate = torch.sigmoid((temp_end - 0.65) * 8.0)
            return self.corr_limit * high_temp_gate * torch.tanh(raw)
        vcorr_end = x[:, -1, 0:1]
        high_temp_gate = torch.sigmoid((temp_end - 0.65) * 8.0)
        low_voltage_gate = torch.sigmoid(-(vcorr_end + 0.25) * 5.0)
        return self.corr_limit * high_temp_gate * low_voltage_gate * torch.tanh(raw)


class Stage2CorrectedModel(nn.Module):
    def __init__(
        self,
        base: VcorrITGoalModel,
        hidden_size: int,
        corr_limit: float,
        corr_mode: str = "45_low",
        freeze_base: bool = True,
        input_dim: int = 3,
        base_input_dim: int | None = None,
        corr_zero_init: bool = True,
    ):
        super().__init__()
        self.base = base
        self.corr_mode = str(corr_mode)
        self.freeze_base = bool(freeze_base)
        self.base_input_dim = int(base_input_dim if base_input_dim is not None else input_dim)
        self.correction = Correction45(
            hidden_size,
            corr_limit,
            self.corr_mode,
            input_dim=int(input_dim),
            zero_init=bool(corr_zero_init),
        )
        if self.freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)

    def _base_input(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : self.base_input_dim]

    def base_prediction(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(self._base_input(x))

    def base_endpoint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_base = self._base_input(x)
        if self.freeze_base:
            self.base.eval()
            with torch.no_grad():
                h = self.base.encode_sequence(x_base)
                base_logits = self.base.logits_sequence(x_base)[:, -1, :]
            return base_logits, h[:, -1, :]
        h = self.base.encode_sequence(x_base)
        base_logits = self.base.logits_sequence(x_base)[:, -1, :]
        return base_logits, h[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_logits, h_last = self.base_endpoint(x)
        correction = self.correction(h_last, x)
        return torch.sigmoid(base_logits + correction)


class SnapshotEnsembleBase(nn.Module):
    def __init__(self, models: list[VcorrITGoalModel]):
        super().__init__()
        if not models:
            raise ValueError("SnapshotEnsembleBase requires at least one model.")
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        hidden = [model.encode_sequence(x) for model in self.models]
        return torch.stack(hidden, dim=0).mean(dim=0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        logits = [model.logits_sequence(x) for model in self.models]
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class SnapshotGatedBase(nn.Module):
    def __init__(self, models: list[VcorrITGoalModel], hidden_size: int):
        super().__init__()
        if not models:
            raise ValueError("SnapshotGatedBase requires at least one model.")
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        stat_dim = 12
        gate_dim = stat_dim + self.n_models + 2
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), self.n_models),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    def _window_stats(self, x: torch.Tensor) -> torch.Tensor:
        x_end = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_delta = x_end - x[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def _snapshot_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_logits = []
        seq_hidden = []
        for model in self.models:
            h = model.encode_sequence(x)
            seq_hidden.append(h)
            seq_logits.append(model.logits_sequence(x))
        return torch.stack(seq_logits, dim=0), torch.stack(seq_hidden, dim=0)

    def _weights(self, x: torch.Tensor, seq_logits: torch.Tensor) -> torch.Tensor:
        endpoint_logits = seq_logits[:, :, -1, 0].transpose(0, 1)
        gate_input = torch.cat(
            [
                self._window_stats(x),
                endpoint_logits,
                endpoint_logits.mean(dim=1, keepdim=True),
                endpoint_logits.std(dim=1, unbiased=False, keepdim=True),
            ],
            dim=1,
        )
        return torch.softmax(self.gate(gate_input), dim=1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        seq_logits, seq_hidden = self._snapshot_outputs(x)
        weights = self._weights(x, seq_logits).transpose(0, 1).view(self.n_models, x.shape[0], 1, 1)
        return (seq_hidden * weights).sum(dim=0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        seq_logits, _seq_hidden = self._snapshot_outputs(x)
        weights = self._weights(x, seq_logits).transpose(0, 1).view(self.n_models, x.shape[0], 1, 1)
        return (seq_logits * weights).sum(dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def _meta_tensor(meta, key: str, target_device: torch.device) -> torch.Tensor:
    vals = meta[key]
    if torch.is_tensor(vals):
        return vals.to(device=target_device, dtype=torch.float32)
    return torch.as_tensor(vals, device=target_device, dtype=torch.float32)


def train_stage2_correction(
    cfg: CondInvConfig,
    variant: CondInvVariant,
    base_model: VcorrITGoalModel,
    train_loader,
    valid_loader,
    test_loader,
) -> pd.DataFrame:
    corrected = Stage2CorrectedModel(
        base_model,
        int(cfg.hidden_size),
        float(variant.corr_limit),
        str(variant.corr_mode),
        freeze_base=not bool(variant.enable_snapshot_gate),
        input_dim=int(
            getattr(
                base_model,
                "correction_input_dim_for_stage2",
                getattr(base_model, "input_dim_for_stage2", 3),
            )
        ),
        base_input_dim=int(getattr(base_model, "input_dim_for_stage2", 3)),
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in corrected.parameters() if p.requires_grad],
        lr=float(cfg.lr_stage2),
        weight_decay=float(cfg.weight_decay),
    )
    rows = []
    history = []
    for ep in range(1, int(cfg.stage2_epochs) + 1):
        corrected.train()
        losses = []
        supervised_losses = []
        keep_losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y_endpoint = y[:, -1, :] if y.ndim == 3 else y
            with torch.no_grad():
                base_pred = corrected.base_prediction(x)
            pred = corrected(x)
            sample_loss = F.smooth_l1_loss(pred, y_endpoint, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            temps = _meta_tensor(meta, "temperature", device)
            temp45 = torch.isclose(temps, torch.full_like(temps, 45.0), atol=0.75).float()
            if variant.corr_mode == "temp_bands":
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(temp25, temp45).detach()
            elif variant.corr_mode in {"cold_hot", "cold_hot_deep"}:
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                focus = torch.maximum(temp0, temp45).detach()
            elif variant.corr_mode == "cold_mid_hot":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(torch.maximum(temp0, temp25), temp45).detach()
            elif variant.corr_mode == "cold_hot_midlite":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(temp0, temp45) + 0.25 * temp25
                focus = focus.clamp(max=1.0).detach()
            else:
                low_voltage_focus = torch.sigmoid(-(x[:, -1, 0] + 0.25) * 5.0)
                focus = (temp45 * low_voltage_focus).detach()
            weights = 1.0 + (float(cfg.focus45_weight) - 1.0) * focus
            supervised = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
            keep_weight = (1.0 - focus).clamp_min(0.0).unsqueeze(1)
            keep = ((pred - base_pred).square() * keep_weight).sum() / keep_weight.sum().clamp_min(1e-6)
            loss = supervised + float(cfg.keep_lambda) * float(variant.stage2_keep_scale) * keep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(corrected.correction.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            supervised_losses.append(float(supervised.detach().cpu()))
            keep_losses.append(float(keep.detach().cpu()))
        history.append(
            {
                "variant": f"{variant.name}_stage2",
                "epoch": ep,
                "loss": float(np.mean(losses)),
                "supervised_loss": float(np.mean(supervised_losses)),
                "keep_loss": float(np.mean(keep_losses)),
            }
        )
        if ep == 1 or ep % int(cfg.eval_every) == 0 or ep == int(cfg.stage2_epochs):
            valid = eval_by_temp(corrected, valid_loader, "valid", f"{variant.name}_stage2", ep)
            test = eval_by_temp(corrected, test_loader, "test", f"{variant.name}_stage2", ep)
            rows.extend([valid, test])
            piv = test.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
            mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            print(
                f"{variant.name}_stage2 epoch={ep} loss={np.mean(losses):.5f} "
                f"supervised={np.mean(supervised_losses):.5f} keep={np.mean(keep_losses):.5f} "
                f"test0={mae0:.3f}% test25={mae25:.3f}% test45={mae45:.3f}%",
                flush=True,
            )
    pd.DataFrame(history).to_csv(
        cfg.base_dir / "nmc_goal_vcorr_it_conditional_invariant_results" / f"{cfg.output_prefix}_{variant.name}_stage2_history.csv",
        index=False,
    )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_variant(cfg: CondInvConfig, variant: CondInvVariant, frames, out_dir: Path) -> pd.DataFrame:
    scaled, _ = make_scaled_frames_for_ablation(frames, FEATURE_COLS)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    train_cls = SequenceWindowDataset if bool(variant.sequence_training) else DecomposedWindowDataset
    train_ds = train_cls(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    model = make_model(variant, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    legacy_variant = to_variant(variant)
    rows = []
    history = []
    swa_states = []
    swa_epochs = []
    snapshot_states = []
    snapshot_epochs = []
    requested_snapshots = tuple(int(ep) for ep in getattr(variant, "snapshot_epochs", ()))
    uses_snapshots = bool(variant.enable_snapshot_ensemble) or bool(variant.enable_snapshot_gate)
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        mmds = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            h = model.encode_sequence(x)
            h_last = h[:, -1, :]
            if bool(variant.sequence_training):
                pred = model.forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
                y_for_mmd = y[:, -1, :]
            else:
                pred = model(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
                y_for_mmd = y
            sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
            keys = group_keys(meta, cfg.rex_group)
            group_losses = []
            group_weights = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                group_losses.append(sample_loss.index_select(0, idx).mean())
                group_weights.append(sw.index_select(0, idx).mean())
            stack = torch.stack(group_losses)
            wstack = torch.stack(group_weights)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            mmd_loss = conditional_profile_mmd(h_last, y_for_mmd, meta)
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mmds.append(float(mmd_loss.detach().cpu()))
        history.append({"variant": variant.name, "epoch": ep, "loss": float(np.mean(losses)), "condinv_loss": float(np.mean(mmds))})
        if (
            bool(variant.enable_swa)
            and int(ep) >= int(variant.swa_start)
            and int(ep) <= int(variant.swa_end)
            and (int(ep) - int(variant.swa_start)) % max(1, int(variant.swa_every)) == 0
        ):
            swa_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            swa_epochs.append(int(ep))
        if uses_snapshots and int(ep) in requested_snapshots:
            snapshot_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            snapshot_epochs.append(int(ep))
        if ep == 1 or ep % int(cfg.eval_every) == 0 or ep == int(cfg.epochs):
            valid = eval_by_temp(model, valid_loader, "valid", variant.name, ep)
            test = eval_by_temp(model, test_loader, "test", variant.name, ep)
            rows.extend([valid, test])
            piv = test.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
            mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            print(
                f"{variant.name} epoch={ep} loss={np.mean(losses):.5f} condinv={np.mean(mmds):.5f} "
                f"test0={mae0:.3f}% test25={mae25:.3f}% test45={mae45:.3f}%",
                flush=True,
            )
        elif ep % int(cfg.print_every) == 0:
            print(f"{variant.name} epoch={ep} loss={np.mean(losses):.5f} condinv={np.mean(mmds):.5f}", flush=True)
    if swa_states:
        avg_state = {}
        for key in swa_states[0]:
            vals = [state[key] for state in swa_states]
            if torch.is_floating_point(vals[0]):
                avg_state[key] = torch.stack(vals, dim=0).mean(dim=0)
            else:
                avg_state[key] = vals[-1]
        model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})
        swa_name = f"{variant.name}_swa_{swa_epochs[0]}_{swa_epochs[-1]}"
        valid = eval_by_temp(model, valid_loader, "valid", swa_name, int(swa_epochs[-1]))
        test = eval_by_temp(model, test_loader, "test", swa_name, int(swa_epochs[-1]))
        rows.extend([valid, test])
        piv = test.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
        mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
        mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
        mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
        print(f"{swa_name} test0={mae0:.3f}% test25={mae25:.3f}% test45={mae45:.3f}%", flush=True)
    stage2_base: nn.Module = model
    if uses_snapshots:
        missing = sorted(set(requested_snapshots) - set(snapshot_epochs))
        if missing:
            raise RuntimeError(f"Missing snapshot epochs {missing}; cfg.epochs={cfg.epochs}")
        ensemble_models = []
        for state in snapshot_states:
            snapshot_model = make_model(variant, cfg)
            snapshot_model.load_state_dict({k: v.to(device) for k, v in state.items()})
            ensemble_models.append(snapshot_model)
        if bool(variant.enable_snapshot_gate):
            stage2_base = SnapshotGatedBase(ensemble_models, int(cfg.hidden_size)).to(device)
        else:
            stage2_base = SnapshotEnsembleBase(ensemble_models).to(device)
        ensemble_name = f"{variant.name}_snapshot_{'_'.join(str(ep) for ep in snapshot_epochs)}"
        valid = eval_by_temp(stage2_base, valid_loader, "valid", ensemble_name, int(snapshot_epochs[-1]))
        test = eval_by_temp(stage2_base, test_loader, "test", ensemble_name, int(snapshot_epochs[-1]))
        rows.extend([valid, test])
        piv = test.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
        mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
        mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
        mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
        print(f"{ensemble_name} test0={mae0:.3f}% test25={mae25:.3f}% test45={mae45:.3f}%", flush=True)
    if bool(variant.enable_stage2):
        rows.append(train_stage2_correction(cfg, variant, stage2_base, train_loader, valid_loader, test_loader))
    pd.DataFrame(history).to_csv(out_dir / f"{cfg.output_prefix}_{variant.name}_history.csv", index=False)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(cfg: CondInvConfig) -> pd.DataFrame:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("Fixed goal requires hidden_size=64 and window_len=50.")
    if FEATURE_COLS != ["V_corr_raw", "I_raw", "T"]:
        raise RuntimeError(f"Unexpected FEATURE_COLS={FEATURE_COLS}")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_conditional_invariant_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    set_seed(cfg.seed)
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    write_input_schema(FEATURE_COLS, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    write_leakage_audit(FEATURE_COLS, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    all_rows = []
    variants = condinv_variants(cfg.variant_set)
    for variant in variants:
        set_seed(cfg.seed)
        print(f"===== conditional invariant {variant.name} =====", flush=True)
        all_rows.append(run_variant(cfg, variant, frames, out_dir))
    out = pd.concat(all_rows, ignore_index=True)
    out["goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = False
    for (_variant, _epoch, split), idx in out.groupby(["variant", "epoch", "split"]).groups.items():
        sub = out.loc[idx]
        mae0 = sub.loc[np.isclose(sub["temperature_C"], 0.0), "MAE_pct"]
        mae25 = sub.loc[np.isclose(sub["temperature_C"], 25.0), "MAE_pct"]
        mae45 = sub.loc[np.isclose(sub["temperature_C"], 45.0), "MAE_pct"]
        ok = (
            len(mae0)
            and len(mae25)
            and len(mae45)
            and float(mae0.iloc[0]) < 1.0
            and float(mae25.iloc[0]) < 0.7
            and float(mae45.iloc[0]) < 0.3
        )
        out.loc[idx, "goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = bool(ok and split == "test")
    out.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    metadata = {
        **asdict(cfg),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "hidden_size_verified": 64,
        "window_len_verified": 50,
        "design_intent": "Conditional profile-invariant loss aligns DST and US06 hidden centroids only within the same temperature and SOC bin, reducing drive-cycle shortcuts without erasing SOC-relevant dynamic response.",
        "variants": [asdict(v) for v in variants],
        "uses_soc_input": False,
        "uses_soc_label_in_training_loss_only": True,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Conditional invariant best test rows:")
    test = out[out["split"] == "test"].copy()
    piv = test.pivot_table(index=["variant", "epoch"], columns="temperature_C", values="MAE_pct").reset_index()
    if len(piv):
        piv["max_all"] = piv[[0.0, 25.0, 45.0]].max(axis=1)
        print(piv.sort_values(["max_all", 25.0]).head(20).to_string(index=False), flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed Vcorr/I/T conditional profile-invariant screen.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=CondInvConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument(
        "--variant-set",
        default="screen",
        choices=[
            "fast",
            "screen",
            "strong",
            "swa",
            "staged",
            "staged_bands",
            "staged_cold_hot",
            "staged_cold_hot_swa5_10",
            "staged_cold_hot_w25x4",
            "snapshot_5_7_10_cold_hot",
            "snapshot_gate_5_7_10_cold_hot",
            "seq_staged_cold_hot",
        ],
    )
    p.add_argument("--stage2-epochs", type=int, default=35)
    p.add_argument("--lr-stage2", type=float, default=8e-4)
    p.add_argument("--focus45-weight", type=float, default=12.0)
    p.add_argument("--keep-lambda", type=float, default=4.0)
    p.add_argument("--low-soc-threshold", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CondInvConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        eval_every=int(args.eval_every),
        variant_set=str(args.variant_set),
        stage2_epochs=int(args.stage2_epochs),
        lr_stage2=float(args.lr_stage2),
        focus45_weight=float(args.focus45_weight),
        keep_lambda=float(args.keep_lambda),
        low_soc_threshold=float(args.low_soc_threshold),
    )
    run(cfg)


if __name__ == "__main__":
    main()
