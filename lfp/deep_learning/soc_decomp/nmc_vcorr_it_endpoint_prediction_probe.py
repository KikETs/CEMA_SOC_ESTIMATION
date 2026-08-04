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
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_goal_remote_screen import Variant, VcorrITGoalModel, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_endpoint_probe_seed2"


@dataclass
class ProbeConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 2
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 20
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    record_epochs: tuple[int, ...] = (6, 8, 9, 10, 12, 14, 18)
    variant_set: str = "probe"
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def probe_variants(variant_set: str = "probe") -> list[Variant]:
    base = [
        Variant(
            "probe_tcn5_w0x4_w25x2p2",
            recurrent="tcn",
            layers=5,
            kernel_size=5,
            head_kind="linear",
            temp_mode="moe",
            dropout=0.06,
            lambda_rex=2.0,
            weight_0=4.0,
            weight_25=2.2,
        ),
        Variant(
            "probe_tcn5_w0x3p8_w25x2",
            recurrent="tcn",
            layers=5,
            kernel_size=5,
            head_kind="linear",
            temp_mode="moe",
            dropout=0.06,
            lambda_rex=2.0,
            weight_0=3.8,
            weight_25=2.0,
        ),
        Variant(
            "probe_tcn5_w0x4_w25x2p4",
            recurrent="tcn",
            layers=5,
            kernel_size=5,
            head_kind="linear",
            temp_mode="moe",
            dropout=0.06,
            lambda_rex=2.0,
            weight_0=4.0,
            weight_25=2.4,
        ),
        Variant(
            "probe_tcn6k5_w0x4_w25x2p2",
            recurrent="tcn",
            layers=6,
            kernel_size=5,
            head_kind="linear",
            temp_mode="moe",
            dropout=0.06,
            lambda_rex=2.0,
            weight_0=4.0,
            weight_25=2.2,
        ),
    ]
    if variant_set == "probe_plus":
        base.extend(
            [
                Variant(
                    "probe_tcn5_w0x4p5_w25x2p2",
                    recurrent="tcn",
                    layers=5,
                    kernel_size=5,
                    head_kind="linear",
                    temp_mode="moe",
                    dropout=0.06,
                    lambda_rex=2.0,
                    weight_0=4.5,
                    weight_25=2.2,
                ),
                Variant(
                    "probe_tcn4k3_w0x4_w25x2p2",
                    recurrent="tcn",
                    layers=4,
                    kernel_size=3,
                    head_kind="linear",
                    temp_mode="moe",
                    dropout=0.06,
                    lambda_rex=2.0,
                    weight_0=4.0,
                    weight_25=2.2,
                ),
            ]
        )
    return base


def make_model(variant: Variant, cfg: ProbeConfig) -> nn.Module:
    return VcorrITGoalModel(
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


def metrics_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
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
                "RMSE_pct": float(np.sqrt(np.mean(err ** 2)) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def train_probe_variant(cfg: ProbeConfig, variant: Variant, frames, out_dir: Path) -> pd.DataFrame:
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
    model = make_model(variant, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    predictions = []
    record_epochs = set(int(e) for e in cfg.record_epochs)
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred = model(x)
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            sw = temp_weights(meta, variant, int(sample_loss.numel()))
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
        if ep in record_epochs:
            valid_pred = predict_rows(model, valid_loader, "valid", variant.name, ep)
            test_pred = predict_rows(model, test_loader, "test", variant.name, ep)
            predictions.extend([valid_pred, test_pred])
            metrics = metrics_from_predictions(test_pred)
            piv = metrics.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
            mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            print(f"{variant.name} epoch={ep} loss={np.mean(losses):.5f} test0={mae0:.3f}% test25={mae25:.3f}%", flush=True)
    out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if len(out):
        out["seed"] = int(cfg.seed)
        out["input_feature_dim"] = len(FEATURE_COLS)
        out.to_csv(out_dir / f"{cfg.output_prefix}_{variant.name}_prediction_rows.csv.gz", index=False, compression="gzip")
    return out


def run(cfg: ProbeConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("Fixed goal requires hidden_size=64 and window_len=50.")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_endpoint_probe_results"
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
    preds = []
    variants = probe_variants(cfg.variant_set)
    for variant in variants:
        set_seed(cfg.seed)
        print(f"===== probe {variant.name} =====", flush=True)
        preds.append(train_probe_variant(cfg, variant, frames, out_dir))
    pred = pd.concat([p for p in preds if len(p)], ignore_index=True) if preds else pd.DataFrame()
    metrics = metrics_from_predictions(pred) if len(pred) else pd.DataFrame()
    if len(metrics):
        metrics["goal_0C_lt1_25C_lt0p7"] = False
        for (_variant, _epoch, split), idx in metrics.groupby(["variant", "epoch", "split"]).groups.items():
            sub = metrics.loc[idx]
            mae0 = sub.loc[np.isclose(sub["temperature_C"], 0.0), "MAE_pct"]
            mae25 = sub.loc[np.isclose(sub["temperature_C"], 25.0), "MAE_pct"]
            ok = len(mae0) and len(mae25) and float(mae0.iloc[0]) < 1.0 and float(mae25.iloc[0]) < 0.7
            metrics.loc[idx, "goal_0C_lt1_25C_lt0p7"] = bool(ok and split == "test")
        metrics.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "variants": [asdict(v) for v in variants],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if len(metrics):
        print("Probe best test rows:")
        test = metrics[metrics["split"] == "test"].copy()
        piv = test.pivot_table(index=["variant", "epoch"], columns="temperature_C", values="MAE_pct").reset_index()
        if len(piv):
            piv["max_0_25"] = piv[[0.0, 25.0]].max(axis=1)
            print(piv.sort_values(["max_0_25", 25.0]).head(20).to_string(index=False), flush=True)
    print(f"Schema: {schema.to_dict(orient='records')}", flush=True)
    print(f"Leakage audit: {leakage.to_dict(orient='records')}", flush=True)
    return {"predictions": pred, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save endpoint TCN prediction rows for candidate checkpoint ensembling.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=ProbeConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--record-epochs", default="6,8,9,10,12,14,18")
    p.add_argument("--variant-set", default="probe", choices=["probe", "probe_plus"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProbeConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        record_epochs=tuple(int(s.strip()) for s in str(args.record_epochs).split(",") if s.strip()),
        variant_set=str(args.variant_set),
    )
    run(cfg)


if __name__ == "__main__":
    main()
