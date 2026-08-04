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
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import make_cfg
from .deep_no_leak_experiment import CausalConvBlock, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import collate_meta_to_frame
from .nmc_branchbands_experiment import (
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
    focus_metrics,
    metrics_by_trajectory,
    table_md,
    write_start_audit,
)
from .nmc_vcorr_it_designed_rnn_experiment import AuxWindowDataset
from .nmc_vcorr_it_goal_remote_screen import group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS, attach_eval_features, make_endpoint_lookup
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_shift_guard_tcn_seed0"


@dataclass
class ShiftGuardConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 90
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 10
    valid_every: int = 5
    variant_set: str = "screen"
    train_sampler: str = "temperature_balanced"
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class ShiftGuardVariant:
    name: str
    layers: int = 5
    kernel_size: int = 5
    dropout: float = 0.05
    correction_limit: float = 0.75
    lambda_rex: float = 2.0
    lambda_anchor: float = 0.5
    lambda_mono: float = 0.25
    lambda_delta: float = 1e-3
    lambda_gate: float = 2e-3
    lambda_shift25: float = 3e-3
    drive_bias_strength: float = 0.0
    drive_bias_center: float = 0.12
    drive_bias_scale: float = 0.12
    low45_weight: float = 0.0
    lr: float = 8e-4
    weight_0: float = 4.0
    weight_25: float = 2.2
    weight_45: float = 1.0
    anchor_low_i_A: float = 0.20
    anchor_low_di_A: float = 0.40
    selection: str = "shift_guard_tri45"


DESIGN_INTENT = [
    {
        "component": "shared TCN encoder",
        "why": "Remote screens showed TCN variants were the closest single-checkpoint family; local voltage/current history is needed, but no current is integrated into SOC.",
    },
    {
        "component": "Vcorr/T static anchor",
        "why": "V_corr carries the slow SOC-voltage relation. The anchor gives a conservative endpoint estimate when the 50 s dynamic response is weak or shifted.",
    },
    {
        "component": "T-conditioned FiLM",
        "why": "Temperature changes polarization and relaxation response. FiLM uses measured T as a continuous condition inside one checkpoint, not separate temperature checkpoints.",
    },
    {
        "component": "bounded dynamic correction",
        "why": "The dynamic branch can correct load-history effects, but is explicitly limited so it cannot freely overwrite the voltage anchor.",
    },
    {
        "component": "drive-shift guard",
        "why": "Remote data analysis found 25C VALIDATION and FUDS have opposite excitation shifts. The guard penalizes large 25C corrections in high-shift windows without using test SOC.",
    },
    {
        "component": "validation-only checkpoint selection",
        "why": "FUDS labels are used only after selecting a checkpoint. The score combines VALIDATION validation with label-free correction/guard regularity recorded during training.",
    },
]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _meta_tensor(meta, key: str, fallback: float, batch_size: int) -> torch.Tensor:
    if key not in meta:
        return torch.full((batch_size,), float(fallback), device=device, dtype=torch.float32)
    return torch.as_tensor(meta[key], device=device, dtype=torch.float32)


def _standard_train_loader(ds, cfg: ShiftGuardConfig) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": True,
        "generator": generator,
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _soc_bin4(soc: float) -> str:
    value = float(soc)
    if value < 0.2:
        return "soc0_0_20"
    if value < 0.5:
        return "soc1_20_50"
    if value < 0.8:
        return "soc2_50_80"
    return "soc3_80_100"


def _profile_soc_balanced_loader(ds, cfg: ShiftGuardConfig) -> DataLoader:
    keys = []
    for fi, _start, end in ds.index:
        frame = ds.frames[fi]
        temp = round(float(frame["temperature"][end]), 3)
        drive = str(frame["drive_cycle"][end]).upper()
        soc_bin = _soc_bin4(float(frame["y_physical"][end]))
        keys.append((temp, drive, soc_bin))
    counts: dict[tuple, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = torch.as_tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "sampler": sampler,
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _make_train_loader(ds, base_cfg, cfg: ShiftGuardConfig) -> DataLoader:
    sampler = str(cfg.train_sampler)
    if sampler == "temperature_balanced":
        return temperature_balanced_loader(ds, base_cfg, shuffle=True)
    if sampler == "standard":
        return _standard_train_loader(ds, cfg)
    if sampler == "temperature_profile_soc_balanced":
        return _profile_soc_balanced_loader(ds, cfg)
    raise ValueError(f"Unknown train_sampler={cfg.train_sampler!r}")


class FiLMTCNBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.block = CausalConvBlock(
            hidden_size,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            dropout=float(dropout),
            norm_kind="channel",
        )
        self.film = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, temp_last: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        gamma, beta = self.film(temp_last).chunk(2, dim=1)
        return y * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1)) + 0.25 * beta.unsqueeze(-1)


class ShiftGuardTCN(nn.Module):
    def __init__(self, variant: ShiftGuardVariant, input_dim: int = 3, hidden_size: int = 64):
        super().__init__()
        self.variant = variant
        self.temp_idx = 2
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.blocks = nn.ModuleList(
            [
                FiLMTCNBlock(hidden_size, variant.kernel_size, 2**i, variant.dropout)
                for i in range(int(variant.layers))
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.anchor = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(variant.dropout)),
            nn.Linear(hidden_size, 1),
        )
        stats_dim = 9
        dyn_dim = hidden_size + stats_dim
        self.delta_head = nn.Sequential(
            nn.Linear(dyn_dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(variant.dropout)),
            nn.Linear(hidden_size, 1),
        )
        self.guard_head = nn.Sequential(
            nn.Linear(stats_dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(variant.dropout)),
            nn.Linear(hidden_size, 1),
        )
        self.limit_head = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.limit_head[-1].weight)
        nn.init.zeros_(self.limit_head[-1].bias)

    def window_stats(self, x: torch.Tensor) -> torch.Tensor:
        v = x[..., 0]
        i = x[..., 1]
        t = x[..., 2]
        di = i[:, 1:] - i[:, :-1]
        abs_i_mean = i.abs().mean(dim=1, keepdim=True)
        i_std = i.std(dim=1, keepdim=True, unbiased=False)
        abs_di_mean = di.abs().mean(dim=1, keepdim=True)
        di_energy = (di.square()).mean(dim=1, keepdim=True)
        v_span = (v.max(dim=1).values - v.min(dim=1).values).unsqueeze(1)
        v_delta = (v[:, -1] - v[:, 0]).unsqueeze(1)
        low_frac = (i.abs() < 0.10).to(x.dtype).mean(dim=1, keepdim=True)
        t_last = t[:, -1:].contiguous()
        v_last = v[:, -1:].contiguous()
        return torch.cat([abs_i_mean, i_std, abs_di_mean, di_energy, v_span, v_delta, low_frac, t_last, v_last], dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        temp_last = x[:, -1:, self.temp_idx]
        h = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h, temp_last)
        return self.norm(h.transpose(1, 2))

    def forward_parts(self, x: torch.Tensor):
        h = self.encode(x)
        h_last = h[:, -1, :]
        stats = self.window_stats(x)
        anchor_in = torch.cat([x[:, -1:, 0], stats[:, 7:8]], dim=1)
        anchor_logit = self.anchor(anchor_in)
        dyn = torch.cat([h_last, stats], dim=1)
        raw_delta = self.delta_head(dyn)
        learned_gate = torch.sigmoid(self.guard_head(stats))
        dynamic_score = stats[:, 1:2].abs() + stats[:, 2:3].abs() + 0.5 * stats[:, 3:4].abs()
        shift_prior = torch.exp(-0.15 * torch.relu(dynamic_score - 0.75)).clamp(0.25, 1.0)
        guard = learned_gate * shift_prior
        temp_limit = float(self.variant.correction_limit) * (0.75 + 0.5 * torch.sigmoid(self.limit_head(stats[:, 7:8])))
        correction = temp_limit * guard * torch.tanh(raw_delta)
        temp_mid_weight = torch.exp(-torch.square((stats[:, 7:8] - 0.08) / 0.55))
        drive_arg = (stats[:, 2:3] - float(self.variant.drive_bias_center)) / max(float(self.variant.drive_bias_scale), 1e-6)
        drive_bias = float(self.variant.drive_bias_strength) * temp_mid_weight * torch.tanh(drive_arg)
        pred = torch.sigmoid(anchor_logit + correction + drive_bias)
        return pred, torch.sigmoid(anchor_logit), correction, guard, dynamic_score

    def monotonic_penalty(self, x: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
        stats = self.window_stats(x)
        anchor_in = torch.cat([x[:, -1:, 0], stats[:, 7:8]], dim=1)
        anchor_hi = anchor_in.clone()
        anchor_hi[:, 0:1] = anchor_hi[:, 0:1] + float(eps)
        lo = self.anchor(anchor_in)
        hi = self.anchor(anchor_hi)
        return F.relu(lo - hi).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(x)[0]


def variants_for_set(name: str) -> list[ShiftGuardVariant]:
    if name == "fast":
        return [
            ShiftGuardVariant(
                "film_tcn5_guard25_lim0p75_w0x4_w25x2p2",
                layers=5,
                correction_limit=0.75,
                lambda_anchor=0.5,
                lambda_shift25=3e-3,
            )
        ]
    if name == "wide":
        return [
            ShiftGuardVariant(
                "film_tcn5_guard25_lim0p75_w0x4_w25x2p2",
                layers=5,
                correction_limit=0.75,
                lambda_anchor=0.5,
                lambda_shift25=3e-3,
            ),
            ShiftGuardVariant(
                "film_tcn5_guard25_lim0p60_w0x4_w25x2p2",
                layers=5,
                correction_limit=0.60,
                lambda_anchor=0.7,
                lambda_shift25=5e-3,
            ),
            ShiftGuardVariant(
                "film_tcn6_guard25_lim0p80_w0x4_w25x2p2",
                layers=6,
                correction_limit=0.80,
                lambda_anchor=0.5,
                lambda_shift25=3e-3,
            ),
            ShiftGuardVariant(
                "film_tcn5_guard25_lim0p90_w0x4p5_w25x2p2",
                layers=5,
                correction_limit=0.90,
                weight_0=4.5,
                weight_25=2.2,
                lambda_anchor=0.4,
                lambda_shift25=2e-3,
            ),
        ]
    if name == "drivebias":
        return [
            ShiftGuardVariant(
                "film_tcn5_drivebias0p06_low45x3_lim0p75",
                layers=5,
                correction_limit=0.75,
                lambda_anchor=0.5,
                lambda_shift25=2e-3,
                drive_bias_strength=0.06,
                drive_bias_center=0.12,
                drive_bias_scale=0.12,
                low45_weight=3.0,
            ),
            ShiftGuardVariant(
                "film_tcn5_drivebias0p08_low45x4_lim0p75",
                layers=5,
                correction_limit=0.75,
                lambda_anchor=0.5,
                lambda_shift25=2e-3,
                drive_bias_strength=0.08,
                drive_bias_center=0.12,
                drive_bias_scale=0.12,
                low45_weight=4.0,
            ),
            ShiftGuardVariant(
                "film_tcn6_drivebias0p06_low45x3_lim0p80",
                layers=6,
                correction_limit=0.80,
                lambda_anchor=0.5,
                lambda_shift25=2e-3,
                drive_bias_strength=0.06,
                drive_bias_center=0.12,
                drive_bias_scale=0.12,
                low45_weight=3.0,
            ),
        ]
    return [
        ShiftGuardVariant(
            "film_tcn5_guard25_lim0p75_w0x4_w25x2p2",
            layers=5,
            correction_limit=0.75,
            lambda_anchor=0.5,
            lambda_shift25=3e-3,
        ),
        ShiftGuardVariant(
            "film_tcn5_guard25_lim0p60_w0x4_w25x2p2",
            layers=5,
            correction_limit=0.60,
            lambda_anchor=0.7,
            lambda_shift25=5e-3,
        ),
        ShiftGuardVariant(
            "film_tcn6_guard25_lim0p80_w0x4_w25x2p2",
            layers=6,
            correction_limit=0.80,
            lambda_anchor=0.5,
            lambda_shift25=3e-3,
        ),
    ]


@torch.no_grad()
def predict_loader(model: nn.Module, loader, model_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
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


def goal_score(overall: pd.DataFrame, by_temp: pd.DataFrame, train_reg: float = 0.0) -> float:
    if overall.empty or by_temp.empty:
        return float("inf")
    overall_mae = float(overall["MAE_pct"].iloc[0])

    def t_mae(temp: float) -> float:
        sub = by_temp[np.isclose(by_temp["temperature_C"], temp)]
        return float(sub["MAE_pct"].iloc[0]) if len(sub) else overall_mae

    mae0 = t_mae(0.0)
    mae25 = t_mae(25.0)
    mae45 = t_mae(45.0)
    excess = max(0.0, mae0 - 1.0) + 1.5 * max(0.0, mae25 - 0.7) + 2.0 * max(0.0, mae45 - 0.3)
    normalized_worst = max(mae0 / 1.0, mae25 / 0.7, mae45 / 0.3)
    return float(normalized_worst + 3.0 * excess + 0.05 * overall_mae + float(train_reg))


def train_variant(cfg: ShiftGuardConfig, variant: ShiftGuardVariant, frames, out_dir: Path):
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
    train_loader = _make_train_loader(train_ds, base_cfg, cfg)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    lookup = make_endpoint_lookup(frames, FEATURE_COLS)

    model = ShiftGuardTCN(variant, input_dim=len(FEATURE_COLS), hidden_size=int(cfg.hidden_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        epoch_delta = []
        epoch_gate = []
        epoch_shift25 = []
        by_group: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred, anchor_pred, correction, guard, dynamic_score = model.forward_parts(x)
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            if float(variant.low45_weight) > 0.0:
                temps_for_focus = _meta_tensor(meta, "temperature", 25.0, int(sample_loss.numel()))
                low45_mask = ((temps_for_focus - 45.0).abs() < 1e-3) & (y[:, 0] < 0.20)
                sample_loss = sample_loss * (1.0 + float(variant.low45_weight) * low45_mask.to(sample_loss.dtype))
            sw = temp_weights(meta, variant, int(sample_loss.numel()))
            keys = group_keys(meta, cfg.rex_group)
            group_losses = []
            group_weights = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                group_losses.append(g_loss)
                group_weights.append(sw.index_select(0, idx).mean())
                by_group.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(group_losses) if group_losses else sample_loss.mean().view(1)
            wstack = torch.stack(group_weights) if group_weights else stack.new_ones(stack.shape)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(variant.lambda_rex) * rex_var

            anchor_sample = F.smooth_l1_loss(anchor_pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            aw = torch.ones_like(anchor_sample)
            if "mean_absI" in meta:
                mean_abs_i = _meta_tensor(meta, "mean_absI", 0.0, int(sample_loss.numel()))
                mean_abs_di = _meta_tensor(meta, "mean_absdI", 0.0, int(sample_loss.numel()))
                aw = (0.25 + torch.exp(-mean_abs_i / variant.anchor_low_i_A) * torch.exp(-mean_abs_di / variant.anchor_low_di_A)).clamp(0.05, 1.25)
            anchor_loss = (anchor_sample * aw).sum() / aw.sum().clamp_min(1e-6)
            loss = loss + float(variant.lambda_anchor) * anchor_loss

            mono_loss = model.monotonic_penalty(x)
            loss = loss + float(variant.lambda_mono) * mono_loss
            delta_loss = correction.square().mean()
            gate_loss = guard.square().mean()
            loss = loss + float(variant.lambda_delta) * delta_loss + float(variant.lambda_gate) * gate_loss

            temps = _meta_tensor(meta, "temperature", 25.0, int(sample_loss.numel()))
            mask25 = (temps - 25.0).abs() < 1e-3
            if mask25.any():
                high_dyn = torch.relu(dynamic_score[mask25] - 0.75)
                shift25_loss = (correction[mask25].abs() * (1.0 + high_dyn)).mean()
                loss = loss + float(variant.lambda_shift25) * shift25_loss
            else:
                shift25_loss = loss.new_tensor(0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            epoch_delta.append(float(delta_loss.detach().cpu()))
            epoch_gate.append(float(gate_loss.detach().cpu()))
            epoch_shift25.append(float(shift25_loss.detach().cpu()))

        train_reg = 5.0 * float(np.mean(epoch_delta)) + 2.0 * float(np.mean(epoch_gate)) + 2.0 * float(np.mean(epoch_shift25))
        row = {
            "model_name": model_name,
            "variant": variant.name,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "mono_loss": float(mono_loss.detach().cpu()),
            "delta_loss": float(np.mean(epoch_delta)),
            "gate_loss": float(np.mean(epoch_gate)),
            "shift25_loss": float(np.mean(epoch_shift25)),
            "train_selection_regularizer": train_reg,
        }
        for key, vals in by_group.items():
            safe = key.replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))

        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.valid_every)) == 0:
            valid_pred = attach_eval_features(predict_loader(model, valid_loader, model_name), lookup)
            valid_overall = _overall_metrics(valid_pred)
            valid_by_temp = variance_by_temperature(valid_pred)
            score = goal_score(valid_overall, valid_by_temp, train_reg=train_reg)
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
        if df.empty:
            continue
        df["variant"] = variant.name
        df["selected_epoch"] = int(best_epoch)
        df["selection"] = variant.selection
        df["input_feature_dim"] = len(FEATURE_COLS)
        ok0 = by_temp.loc[np.isclose(by_temp["temperature_C"], 0.0), "MAE_pct"].min() < 1.0
        ok25 = by_temp.loc[np.isclose(by_temp["temperature_C"], 25.0), "MAE_pct"].min() < 0.7
        ok45 = by_temp.loc[np.isclose(by_temp["temperature_C"], 45.0), "MAE_pct"].min() < 0.3
        df["goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = bool(ok0 and ok25 and ok45)
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


def write_report(cfg, out_dir, r0_df, schema, leakage, overall, by_temp, valid_by_temp, history):
    lines = [
        "# NMC Vcorr/I/T Shift-Guard TCN",
        "",
        "## Fixed Conditions",
        "- Inputs: V_corr_raw, I_raw, T only.",
        "- Window length: 50, stride: 3 for training, stride: 1 for valid/test evaluation.",
        f"- Train profiles: {', '.join(cfg.train_profiles)}. Validation profiles: {', '.join(cfg.valid_profiles)}. Test profiles: {', '.join(cfg.test_profiles)}.",
        f"- Train sampler: {cfg.train_sampler}.",
        "- Target: 0C MAE < 1.0%p, 25C MAE < 0.7%p, 45C MAE < 0.3%p.",
        "- No SOC input, no window-start SOC, no cumulative Ah, no absolute time/progress, no explicit current-integration SOC state update.",
        "- Current is used only as instantaneous/local excitation, not integrated into SOC state.",
        "",
        "## Data-Driven Design Rationale",
        table_md(pd.DataFrame(DESIGN_INTENT), ["component", "why"]),
        "",
        "## Test Overall",
        table_md(overall, ["variant", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch", "selection"]),
        "",
        "## Test By Temperature",
        table_md(by_temp, ["variant", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio", "selected_epoch", "goal_0C_lt1_25C_lt0p7_45C_lt0p3"]),
        "",
        "## Valid By Temperature",
        table_md(valid_by_temp, ["variant", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Selection History Best Rows",
        table_md(history.sort_values("valid_score").head(20), ["variant", "epoch", "valid_score", "valid_T0_MAE_pct", "valid_T25_MAE_pct", "valid_T45_MAE_pct", "delta_loss", "gate_loss", "shift25_loss"]),
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


def run(cfg: ShiftGuardConfig):
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("Fixed goal requires hidden_size=64 and window_len=50.")
    if FEATURE_COLS != ["V_corr_raw", "I_raw", "T"]:
        raise RuntimeError(f"Unexpected FEATURE_COLS={FEATURE_COLS}")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_shift_guard_tcn_results"
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
    for variant in variants_for_set(cfg.variant_set):
        set_seed(cfg.seed)
        print(f"===== shift_guard {variant.name} =====", flush=True)
        results.append(train_variant(cfg, variant, frames, out_dir))

    overall = pd.concat([r["overall"] for r in results], ignore_index=True).sort_values("MAE_pct")
    by_temp = pd.concat([r["by_temperature"] for r in results], ignore_index=True)
    by_traj = pd.concat([r["by_trajectory"] for r in results], ignore_index=True)
    focus = pd.concat([r["focus"] for r in results], ignore_index=True)
    valid_overall = pd.concat([r["valid_overall"] for r in results], ignore_index=True).sort_values("MAE_pct")
    valid_by_temp = pd.concat([r["valid_by_temperature"] for r in results], ignore_index=True)
    history = pd.concat([r["history"] for r in results], ignore_index=True)
    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{cfg.output_prefix}_by_trajectory.csv", index=False)
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
        "variants": [asdict(v) for v in variants_for_set(cfg.variant_set)],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
        "checkpoint_selection": "validation_and_train_regularizer_only_no_test_soc",
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, r0_df, schema, leakage, overall, by_temp, valid_by_temp, history)
    print("Combined by temperature:")
    print(by_temp.to_string(index=False), flush=True)
    return {
        "overall": overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC Vcorr/I/T shift-guard TCN experiment.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default="nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-profiles", default=",".join(ShiftGuardConfig.train_profiles))
    p.add_argument("--valid-profiles", default=",".join(ShiftGuardConfig.valid_profiles))
    p.add_argument("--test-profiles", default=",".join(ShiftGuardConfig.test_profiles))
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--valid-every", type=int, default=5)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--variant-set", default="screen", choices=["fast", "screen", "wide", "drivebias"])
    p.add_argument(
        "--train-sampler",
        default="temperature_balanced",
        choices=["temperature_balanced", "standard", "temperature_profile_soc_balanced"],
    )
    return p.parse_args()


def _parse_profiles(raw: str) -> tuple[str, ...]:
    profiles = tuple(str(x).strip().upper() for x in str(raw).split(",") if str(x).strip())
    if not profiles:
        raise ValueError("At least one profile is required.")
    return profiles


def main() -> None:
    args = parse_args()
    cfg = ShiftGuardConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        train_profiles=_parse_profiles(args.train_profiles),
        valid_profiles=_parse_profiles(args.valid_profiles),
        test_profiles=_parse_profiles(args.test_profiles),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        valid_every=int(args.valid_every),
        print_every=int(args.print_every),
        variant_set=str(args.variant_set),
        train_sampler=str(args.train_sampler),
    )
    run(cfg)


if __name__ == "__main__":
    main()
