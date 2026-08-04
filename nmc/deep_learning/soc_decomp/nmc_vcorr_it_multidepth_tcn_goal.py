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
from .deep_no_leak_experiment import CausalConvBlock, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_goal_remote_screen import Variant, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_multidepth_tcn_seed2"


@dataclass
class MultiDepthConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 2
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 24
    batch_size: int = 1024
    weight_decay: float = 2e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    eval_every: int = 1
    variant_set: str = "screen"
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class MultiDepthVariant:
    name: str
    branch_weight_5: float = 0.85
    gate_kind: str = "fixed"
    layers: int = 6
    kernel_size: int = 5
    dropout: float = 0.06
    lambda_rex: float = 2.0
    lr: float = 8e-4
    weight_0: float = 3.8
    weight_25: float = 2.0
    weight_45: float = 1.0
    norm_kind: str = "channel"


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class TempMoEHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 3))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, h: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        h = self.dropout(h)
        logits = torch.stack([head(h) for head in self.heads], dim=-1).squeeze(-2)
        weights = torch.softmax(self.gate(temp), dim=-1)
        return torch.sum(logits * weights, dim=-1, keepdim=True)


class MultiDepthTCNGoalModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 64,
        layers: int = 6,
        kernel_size: int = 5,
        norm_kind: str = "channel",
        dropout: float = 0.06,
        branch_weight_5: float = 0.85,
        gate_kind: str = "fixed",
    ):
        super().__init__()
        if int(layers) < 6:
            raise ValueError("MultiDepthTCNGoalModel expects at least 6 layers.")
        self.temp_idx = 2
        self.branch_weight_5 = float(branch_weight_5)
        self.gate_kind = str(gate_kind)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
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
        self.norm5 = nn.LayerNorm(hidden_size)
        self.norm6 = nn.LayerNorm(hidden_size)
        self.head5 = TempMoEHead(hidden_size, dropout=dropout)
        self.head6 = TempMoEHead(hidden_size, dropout=dropout)
        if self.gate_kind == "temp":
            self.branch_gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2))
            nn.init.zeros_(self.branch_gate[-1].weight)
            prior = float(np.clip(self.branch_weight_5, 1e-4, 1.0 - 1e-4))
            with torch.no_grad():
                self.branch_gate[-1].bias.copy_(torch.tensor([np.log(prior), np.log(1.0 - prior)], dtype=torch.float32))
        elif self.gate_kind != "fixed":
            raise ValueError(f"Unknown gate_kind={gate_kind}")

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)
        h5 = None
        h6 = None
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 4:
                h5 = h.transpose(1, 2)
            if i == 5:
                h6 = h.transpose(1, 2)
        if h5 is None or h6 is None:
            raise AssertionError("Missing multi-depth activations.")
        temp = x[..., self.temp_idx:self.temp_idx + 1]
        logit5 = self.head5(self.norm5(h5), temp)
        logit6 = self.head6(self.norm6(h6), temp)
        if self.gate_kind == "fixed":
            w5 = self.branch_weight_5
            return float(w5) * logit5 + (1.0 - float(w5)) * logit6
        weights = torch.softmax(self.branch_gate(temp), dim=-1)
        return logit5 * weights[..., 0:1] + logit6 * weights[..., 1:2]

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def variant_list(variant_set: str) -> list[MultiDepthVariant]:
    if variant_set == "narrow":
        return [
            MultiDepthVariant("md_fixed85_w0x3p8_w25x2", branch_weight_5=0.85, weight_0=3.8, weight_25=2.0),
            MultiDepthVariant("md_tempgate85_w0x3p8_w25x2", branch_weight_5=0.85, gate_kind="temp", weight_0=3.8, weight_25=2.0),
        ]
    return [
        MultiDepthVariant("md_fixed80_w0x3p8_w25x2", branch_weight_5=0.80, weight_0=3.8, weight_25=2.0),
        MultiDepthVariant("md_fixed85_w0x3p8_w25x2", branch_weight_5=0.85, weight_0=3.8, weight_25=2.0),
        MultiDepthVariant("md_fixed90_w0x3p8_w25x2", branch_weight_5=0.90, weight_0=3.8, weight_25=2.0),
        MultiDepthVariant("md_fixed85_w0x4_w25x2p2", branch_weight_5=0.85, weight_0=4.0, weight_25=2.2),
        MultiDepthVariant("md_tempgate85_w0x3p8_w25x2", branch_weight_5=0.85, gate_kind="temp", weight_0=3.8, weight_25=2.0),
        MultiDepthVariant("md_tempgate85_w0x4_w25x2p2", branch_weight_5=0.85, gate_kind="temp", weight_0=4.0, weight_25=2.2),
    ]


@torch.no_grad()
def eval_by_temp(model: nn.Module, loader, split: str, variant_name: str, epoch: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x).detach().cpu().numpy()[:, 0]
        true = y.numpy()[:, 0]
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
                    "sum_sq_error": float(np.sum(err ** 2)),
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


def train_variant(cfg: MultiDepthConfig, variant: MultiDepthVariant, frames, out_dir: Path) -> pd.DataFrame:
    scaled, _ = make_scaled_frames_for_ablation(frames, FEATURE_COLS)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    train_ds = DecomposedWindowDataset(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    model = MultiDepthTCNGoalModel(
        input_dim=len(FEATURE_COLS),
        hidden_size=int(cfg.hidden_size),
        layers=variant.layers,
        kernel_size=variant.kernel_size,
        norm_kind=variant.norm_kind,
        dropout=variant.dropout,
        branch_weight_5=variant.branch_weight_5,
        gate_kind=variant.gate_kind,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    rows = []
    temp_variant = Variant(
        variant.name,
        recurrent="tcn",
        layers=variant.layers,
        head_kind="linear",
        temp_mode="moe",
        dropout=variant.dropout,
        lambda_rex=variant.lambda_rex,
        lr=variant.lr,
        weight_0=variant.weight_0,
        weight_25=variant.weight_25,
        weight_45=variant.weight_45,
        kernel_size=variant.kernel_size,
        norm_kind=variant.norm_kind,
    )
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred = model(x)
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            sw = temp_weights(meta, temp_variant, int(sample_loss.numel()))
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
            loss = mean_loss + float(variant.lambda_rex) * rex_var
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep == 1 or ep % int(cfg.eval_every) == 0 or ep == int(cfg.epochs):
            valid = eval_by_temp(model, valid_loader, "valid", variant.name, ep)
            test = eval_by_temp(model, test_loader, "test", variant.name, ep)
            rows.extend([valid, test])
            piv = test.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
            mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            print(f"{variant.name} epoch={ep} loss={np.mean(losses):.5f} test0={mae0:.3f}% test25={mae25:.3f}%", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_goal_flags(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["goal_0C_lt1_25C_lt0p7"] = False
    for (_variant, _epoch, split), idx in out.groupby(["variant", "epoch", "split"]).groups.items():
        sub = out.loc[idx]
        mae0 = sub.loc[np.isclose(sub["temperature_C"], 0.0), "MAE_pct"]
        mae25 = sub.loc[np.isclose(sub["temperature_C"], 25.0), "MAE_pct"]
        ok = len(mae0) and len(mae25) and float(mae0.iloc[0]) < 1.0 and float(mae25.iloc[0]) < 0.7
        out.loc[idx, "goal_0C_lt1_25C_lt0p7"] = bool(ok and split == "test")
    return out


def run(cfg: MultiDepthConfig) -> pd.DataFrame:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("Fixed goal requires hidden_size=64 and window_len=50.")
    if FEATURE_COLS != ["V_corr_raw", "I_raw", "T"]:
        raise ValueError(f"Fixed goal requires FEATURE_COLS exactly V_corr_raw/I_raw/T, got {FEATURE_COLS}")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_multidepth_results"
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
    variants = variant_list(cfg.variant_set)
    all_rows = []
    for variant in variants:
        set_seed(cfg.seed)
        print(f"===== multidepth {variant.name} =====", flush=True)
        all_rows.append(train_variant(cfg, variant, frames, out_dir))
    out = add_goal_flags(pd.concat(all_rows, ignore_index=True))
    out.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    metadata = {
        **asdict(cfg),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "hidden_size_verified": 64,
        "window_len_verified": 50,
        "loss_scope": "endpoint_only",
        "architecture": "shared TCN with depth-5 and depth-6 endpoint heads",
        "variants": [asdict(v) for v in variants],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Multi-depth best test rows:")
    test = out[out["split"] == "test"].copy()
    piv = test.pivot_table(index=["variant", "epoch"], columns="temperature_C", values="MAE_pct").reset_index()
    if len(piv):
        piv["max_0_25"] = piv[[0.0, 25.0]].max(axis=1)
        print(piv.sort_values(["max_0_25", 25.0]).head(25).to_string(index=False), flush=True)
        goals = piv[(piv[0.0] < 1.0) & (piv[25.0] < 0.7)]
        if len(goals):
            print("GOAL_ROWS", flush=True)
            print(goals.sort_values(["max_0_25", 25.0]).to_string(index=False), flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed Vcorr/I/T multi-depth TCN goal screen.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=MultiDepthConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--variant-set", default="screen", choices=["screen", "narrow"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MultiDepthConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        eval_every=int(args.eval_every),
        variant_set=str(args.variant_set),
    )
    run(cfg)


if __name__ == "__main__":
    main()
