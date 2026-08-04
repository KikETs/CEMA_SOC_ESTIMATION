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
from .models import collate_meta_to_frame
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_designed_rnn_experiment import AuxWindowDataset
from .nmc_vcorr_it_goal_remote_screen import Variant, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_adapter_expert_seed0"


@dataclass
class AdapterConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 70
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
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class AdapterVariant:
    name: str
    layers: int = 5
    kernel_size: int = 5
    dropout: float = 0.05
    lambda_rex: float = 2.0
    focus25_weight: float = 4.0
    focus45_low_weight: float = 4.0
    limit25: float = 0.8
    limit45: float = 0.8
    lambda_delta: float = 2e-4
    weight_0: float = 4.0
    weight_25: float = 2.2
    weight_45: float = 1.0
    lr: float = 8e-4


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def adapter_variants(name: str) -> list[AdapterVariant]:
    if name == "fast":
        return [AdapterVariant("adapter_tcn5_f25x4_l45x4_lim0p8")]
    return [
        AdapterVariant("adapter_tcn5_f25x4_l45x4_lim0p8", focus25_weight=4.0, focus45_low_weight=4.0, limit25=0.8, limit45=0.8),
        AdapterVariant("adapter_tcn5_f25x8_l45x6_lim1p0", focus25_weight=8.0, focus45_low_weight=6.0, limit25=1.0, limit45=1.0),
        AdapterVariant("adapter_tcn5_f25x10_l45x8_lim1p2", focus25_weight=10.0, focus45_low_weight=8.0, limit25=1.2, limit45=1.2),
        AdapterVariant("adapter_tcn6_f25x4_l45x4_lim0p8", layers=6, focus25_weight=4.0, focus45_low_weight=4.0, limit25=0.8, limit45=0.8),
    ]


class AdapterExpertTCN(nn.Module):
    def __init__(self, variant: AdapterVariant, input_dim: int = 3, hidden_size: int = 64):
        super().__init__()
        self.variant = variant
        self.temp_idx = 2
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.blocks = nn.Sequential(
            *[
                CausalConvBlock(
                    hidden_size,
                    kernel_size=int(variant.kernel_size),
                    dilation=2**i,
                    dropout=float(variant.dropout),
                    norm_kind="channel",
                )
                for i in range(int(variant.layers))
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        stats_dim = 8
        in_dim = hidden_size + stats_dim
        self.base_head = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.SiLU(), nn.Dropout(float(variant.dropout)), nn.Linear(hidden_size, 1))
        self.adapter25 = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.SiLU(), nn.Dropout(float(variant.dropout)), nn.Linear(hidden_size, 1))
        self.adapter45 = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.SiLU(), nn.Dropout(float(variant.dropout)), nn.Linear(hidden_size, 1))
        self.gate25 = nn.Sequential(nn.Linear(stats_dim, 16), nn.SiLU(), nn.Linear(16, 1))
        self.gate45 = nn.Sequential(nn.Linear(stats_dim, 16), nn.SiLU(), nn.Linear(16, 1))
        self.temp_bias = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.temp_bias[-1].weight)
        nn.init.zeros_(self.temp_bias[-1].bias)

    def stats(self, x: torch.Tensor) -> torch.Tensor:
        v = x[..., 0]
        i = x[..., 1]
        t = x[..., 2]
        di = i[:, 1:] - i[:, :-1]
        return torch.cat(
            [
                i.abs().mean(dim=1, keepdim=True),
                i.std(dim=1, keepdim=True, unbiased=False),
                di.abs().mean(dim=1, keepdim=True),
                di.square().mean(dim=1, keepdim=True),
                (v.max(dim=1).values - v.min(dim=1).values).unsqueeze(1),
                (v[:, -1] - v[:, 0]).unsqueeze(1),
                v[:, -1:].contiguous(),
                t[:, -1:].contiguous(),
            ],
            dim=1,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)
        h = self.blocks(h).transpose(1, 2)
        return self.norm(h)

    def forward_parts(self, x: torch.Tensor):
        h = self.encode(x)
        h_last = h[:, -1, :]
        stats = self.stats(x)
        z = torch.cat([h_last, stats], dim=1)
        t_last = stats[:, 7:8]
        temp25_prior = torch.exp(-torch.square((t_last - 0.08) / 0.45))
        temp45_prior = torch.exp(-torch.square((t_last - 1.10) / 0.35))
        dynamic_gate = torch.sigmoid((stats[:, 2:3] - 0.45) / 0.35)
        low_v_gate = torch.sigmoid((-stats[:, 6:7] - 0.25) / 0.35)
        g25 = temp25_prior * dynamic_gate * torch.sigmoid(self.gate25(stats))
        g45 = temp45_prior * low_v_gate * torch.sigmoid(self.gate45(stats))
        base = self.base_head(z) + self.temp_bias(t_last)
        d25 = float(self.variant.limit25) * g25 * torch.tanh(self.adapter25(z))
        d45 = float(self.variant.limit45) * g45 * torch.tanh(self.adapter45(z))
        logit = base + d25 + d45
        return torch.sigmoid(logit), d25, d45, g25, g45

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(x)[0]


def to_legacy_variant(v: AdapterVariant) -> Variant:
    return Variant(
        name=v.name,
        recurrent="tcn",
        layers=v.layers,
        head_kind="linear",
        temp_mode="none",
        dropout=v.dropout,
        lambda_rex=v.lambda_rex,
        lr=v.lr,
        weight_0=v.weight_0,
        weight_25=v.weight_25,
        weight_45=v.weight_45,
        kernel_size=v.kernel_size,
        norm_kind="channel",
    )


def _meta_tensor(meta, key: str, fallback: float, batch_size: int) -> torch.Tensor:
    if key not in meta:
        return torch.full((batch_size,), float(fallback), device=device, dtype=torch.float32)
    return torch.as_tensor(meta[key], device=device, dtype=torch.float32)


@torch.no_grad()
def predict_rows(model: nn.Module, loader, split: str, variant_name: str, epoch: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        pred = model(x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
        mdf = collate_meta_to_frame(meta)
        mdf["split"] = split
        mdf["variant"] = variant_name
        mdf["epoch"] = int(epoch)
        mdf["y_true"] = y.numpy()[:, 0]
        mdf["y_pred"] = pred.detach().cpu().numpy()[:, 0]
        rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        if "temperature" in out.columns and "temperature_C" not in out.columns:
            out["temperature_C"] = out["temperature"].astype(float)
    return out


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, epoch, split, temp), g in pred.groupby(["variant", "epoch", "split", "temperature_C"]):
        err = g["y_pred"].to_numpy() - g["y_true"].to_numpy()
        rows.append(
            {
                "variant": variant,
                "epoch": int(epoch),
                "split": split,
                "temperature_C": float(temp),
                "n_windows": int(len(g)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def run_variant(cfg: AdapterConfig, variant: AdapterVariant, frames, out_dir: Path) -> pd.DataFrame:
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
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    model = AdapterExpertTCN(variant, input_dim=len(FEATURE_COLS), hidden_size=int(cfg.hidden_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    legacy = to_legacy_variant(variant)
    all_pred = []
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred, d25, d45, g25, g45 = model.forward_parts(x)
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            bs = int(sample_loss.numel())
            temps = _meta_tensor(meta, "temperature", 25.0, bs)
            mean_abs_di = _meta_tensor(meta, "mean_absdI", 0.0, bs)
            focus25 = ((temps - 25.0).abs() < 1e-3).to(sample_loss.dtype) * torch.sigmoid((mean_abs_di - 0.12) / 0.05)
            focus45 = (((temps - 45.0).abs() < 1e-3) & (y[:, 0] < 0.20)).to(sample_loss.dtype)
            sample_loss = sample_loss * (1.0 + float(variant.focus25_weight) * focus25 + float(variant.focus45_low_weight) * focus45)
            sw = temp_weights(meta, legacy, bs)
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
            delta_loss = d25.square().mean() + d45.square().mean()
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_delta) * delta_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"variant": variant.name, "epoch": ep, "loss": float(np.mean(losses))})
        if ep == 1 or ep % int(cfg.eval_every) == 0 or ep == int(cfg.epochs):
            valid = predict_rows(model, valid_loader, "valid", variant.name, ep)
            test = predict_rows(model, test_loader, "test", variant.name, ep)
            all_pred.extend([valid, test])
            mt = metrics(test)
            piv = mt.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
            mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            print(f"{variant.name} epoch={ep} loss={np.mean(losses):.5f} test0={mae0:.3f}% test25={mae25:.3f}% test45={mae45:.3f}%", flush=True)
        elif ep % int(cfg.print_every) == 0:
            print(f"{variant.name} epoch={ep} loss={np.mean(losses):.5f}", flush=True)
    pd.DataFrame(history).to_csv(out_dir / f"{cfg.output_prefix}_{variant.name}_history.csv", index=False)
    return pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()


def run(cfg: AdapterConfig) -> pd.DataFrame:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("Fixed goal requires hidden_size=64 and window_len=50.")
    if FEATURE_COLS != ["V_corr_raw", "I_raw", "T"]:
        raise RuntimeError(f"Unexpected FEATURE_COLS={FEATURE_COLS}")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_adapter_expert_results"
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
    preds = []
    variants = adapter_variants(cfg.variant_set)
    for variant in variants:
        set_seed(cfg.seed)
        print(f"===== adapter expert {variant.name} =====", flush=True)
        preds.append(run_variant(cfg, variant, frames, out_dir))
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred.to_csv(out_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    by_temp = metrics(pred) if len(pred) else pd.DataFrame()
    if len(by_temp):
        by_temp["goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = False
        for (_variant, _epoch, split), idx in by_temp.groupby(["variant", "epoch", "split"]).groups.items():
            sub = by_temp.loc[idx]
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
            by_temp.loc[idx, "goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = bool(ok and split == "test")
        by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    metadata = {
        **asdict(cfg),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "design_intent": "Single-checkpoint shared TCN with continuous 25C dynamic and 45C low-voltage adapters. Gates use only V_corr/I/T-derived window-local signals.",
        "variants": [asdict(v) for v in variants],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if len(by_temp):
        print("Adapter expert best test rows:")
        test = by_temp[by_temp["split"] == "test"].copy()
        piv = test.pivot_table(index=["variant", "epoch"], columns="temperature_C", values="MAE_pct").reset_index()
        if len(piv):
            piv["max_all"] = piv[[0.0, 25.0, 45.0]].max(axis=1)
            print(piv.sort_values(["max_all", 25.0]).head(30).to_string(index=False), flush=True)
    return by_temp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed Vcorr/I/T adapter expert single-checkpoint screen.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=AdapterConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--variant-set", default="screen", choices=["fast", "screen"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AdapterConfig(
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
