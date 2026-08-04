from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import make_cfg
from .deep_no_leak_experiment import make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_condinv_staged_exact import (
    CondInvConfig,
    CondInvVariant,
    conditional_profile_mmd,
    make_model,
    set_seed,
    to_variant,
)
from .nmc_vcorr_it_goal_remote_screen import group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_train_selector_audit"


@dataclass
class TrainSelectorAuditConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seeds: tuple[int, ...] = (0, 1, 2)
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@torch.no_grad()
def eval_by_temp_drive(model: nn.Module, loader, split: str, variant_name: str, seed: int, epoch: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x).detach().cpu().numpy()[:, 0]
        y_cpu = y.detach().cpu()
        true = y_cpu[:, -1, 0].numpy() if y_cpu.ndim == 3 else y_cpu.numpy()[:, 0]
        temps_src = meta["temperature"]
        drives_src = meta["drive_cycle"]
        temps = temps_src.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(temps_src) else np.asarray(temps_src, dtype=np.float32)
        drives = drives_src.detach().cpu().numpy().astype(str) if torch.is_tensor(drives_src) else np.asarray(drives_src, dtype=str)
        for temp in sorted(set(float(t) for t in temps)):
            for drive in sorted(set(str(d) for d in drives)):
                idx = np.isclose(temps, temp) & (drives == drive)
                if not bool(np.any(idx)):
                    continue
                err = pred[idx] - true[idx]
                rows.append(
                    {
                        "variant": variant_name,
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "split": split,
                        "temperature_C": float(temp),
                        "drive_cycle": str(drive),
                        "n_windows": int(idx.sum()),
                        "sum_abs_error": float(np.sum(np.abs(err))),
                        "sum_sq_error": float(np.sum(err**2)),
                        "sum_error": float(np.sum(err)),
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.groupby(["variant", "seed", "epoch", "split", "temperature_C", "drive_cycle"], as_index=False).agg(
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


def _train_one_seed(cfg: TrainSelectorAuditConfig, frames, out_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    scaled, _ = make_scaled_frames_for_ablation(frames, FEATURE_COLS)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0

    train_ds = DecomposedWindowDataset(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    train_eval_ds = DecomposedWindowDataset(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    train_eval_loader = make_eval_loader(train_eval_ds, cfg)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)

    variant = CondInvVariant("condinv_tcn5_mmd0p02_train_selector", lambda_condinv=0.02)
    model = make_model(variant, CondInvConfig(hidden_size=cfg.hidden_size, window_len=cfg.window_len))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    legacy_variant = to_variant(variant)
    metric_rows = []
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        mmds = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            h = model.encode_sequence(x)
            pred = model(x)
            h_last = h[:, -1, :]
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
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
            mmd_loss = conditional_profile_mmd(h_last, y, meta)
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mmds.append(float(mmd_loss.detach().cpu()))
        history.append(
            {
                "variant": variant.name,
                "seed": int(seed),
                "epoch": int(ep),
                "loss": float(np.mean(losses)),
                "condinv_loss": float(np.mean(mmds)),
            }
        )
        metric_rows.extend(
            [
                eval_by_temp_drive(model, train_eval_loader, "train", variant.name, seed, ep),
                eval_by_temp_drive(model, valid_loader, "valid", variant.name, seed, ep),
                eval_by_temp_drive(model, test_loader, "test", variant.name, seed, ep),
            ]
        )
        test = metric_rows[-1]
        piv = test.groupby("temperature_C")["MAE_pct"].mean()
        print(
            f"seed={seed} epoch={ep} loss={np.mean(losses):.5f} "
            f"test0={piv.get(0.0, np.nan):.3f}% test25={piv.get(25.0, np.nan):.3f}% test45={piv.get(45.0, np.nan):.3f}%",
            flush=True,
        )
    return pd.concat(metric_rows, ignore_index=True), pd.DataFrame(history)


def summarize_selectors(metrics: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, sdf in metrics.groupby("seed"):
        train = sdf[sdf["split"] == "train"].copy()
        test = sdf[sdf["split"] == "test"].copy()
        valid = sdf[sdf["split"] == "valid"].copy()
        train_piv = train.pivot_table(index="epoch", columns=["temperature_C", "drive_cycle"], values="MAE_pct", aggfunc="mean")
        test_piv = test.pivot_table(index="epoch", columns="temperature_C", values="MAE_pct", aggfunc="mean")
        valid_piv = valid.pivot_table(index="epoch", columns="temperature_C", values="MAE_pct", aggfunc="mean")
        h = history[history["seed"] == seed].set_index("epoch")
        candidate_scores = {}
        if (25.0, "DST") in train_piv.columns and (25.0, "US06") in train_piv.columns:
            dst = train_piv[(25.0, "DST")]
            us06 = train_piv[(25.0, "US06")]
            candidate_scores["min_train25_avg"] = (dst + us06) / 2.0
            candidate_scores["min_train25_worst"] = pd.concat([dst, us06], axis=1).max(axis=1)
            candidate_scores["min_train25_gap"] = (dst - us06).abs()
        if len(train_piv.columns):
            candidate_scores["min_train_all_worst"] = train_piv.max(axis=1)
        if len(valid_piv.columns):
            candidate_scores["min_valid_all_worst"] = valid_piv.max(axis=1)
        if "loss" in h:
            candidate_scores["min_train_loss"] = h["loss"]
        if "condinv_loss" in h:
            candidate_scores["min_condinv_loss"] = h["condinv_loss"]
        for selector, score in candidate_scores.items():
            score = score.dropna()
            if not len(score):
                continue
            ep = int(score.idxmin())
            test0 = float(test_piv.loc[ep].get(0.0, np.nan))
            test25 = float(test_piv.loc[ep].get(25.0, np.nan))
            test45 = float(test_piv.loc[ep].get(45.0, np.nan))
            rows.append(
                {
                    "seed": int(seed),
                    "selector": selector,
                    "selected_epoch": ep,
                    "selector_score": float(score.loc[ep]),
                    "test_0C_MAE": test0,
                    "test_25C_MAE": test25,
                    "test_45C_MAE": test45,
                    "test_max_MAE": float(np.nanmax([test0, test25, test45])),
                    "stage1_25C_good": bool(test25 < 0.7),
                }
            )
    return pd.DataFrame(rows)


def run(cfg: TrainSelectorAuditConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_selector_audit_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    write_input_schema(FEATURE_COLS, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    write_leakage_audit(FEATURE_COLS, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    metric_parts = []
    history_parts = []
    for seed in cfg.seeds:
        metrics, history = _train_one_seed(cfg, frames, out_dir, int(seed))
        metric_parts.append(metrics)
        history_parts.append(history)
    metrics = pd.concat(metric_parts, ignore_index=True)
    history = pd.concat(history_parts, ignore_index=True)
    selectors = summarize_selectors(metrics, history)
    metrics.to_csv(out_dir / f"{cfg.output_prefix}_by_group.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    selectors.to_csv(out_dir / f"{cfg.output_prefix}_selector_summary.csv", index=False)
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(
        json.dumps({**asdict(cfg), "feature_columns": FEATURE_COLS}, indent=2, default=str),
        encoding="utf-8",
    )
    print("Selector summary:")
    print(selectors.sort_values(["selector", "seed"]).to_string(index=False), flush=True)
    return metrics, history, selectors


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-only selector audit for NMC Vcorr/I/T conditional TCN.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=TrainSelectorAuditConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainSelectorAuditConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seeds=_parse_seeds(args.seeds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    run(cfg)


if __name__ == "__main__":
    main()
