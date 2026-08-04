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
from .nmc_vit_feature_lstm_experiment import (
    add_vit_engineered_features,
    attach_eval_features,
    make_endpoint_lookup,
    write_input_schema,
    write_leakage_audit,
)
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


FEATURE_COLS = ["V_corr_raw", "I_raw", "T"]
BASE_PREFIX = "nmc_vcorr_it_lstm_singlehead_l1_h64_w50_bytemp_seed0"


@dataclass
class NMCVcorrITSingleHeadByTempConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    temperatures: tuple[float, ...] = (0.0, 25.0, 45.0)
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 64
    layers: int = 1
    lambda_rex: float = 2.0
    rex_group: str = "drive"
    huber_beta: float = 0.02
    sequence_loss: bool = True
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 10
    valid_every: int = 10
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


class SingleHeadStatelessLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_size: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        out, _ = self.lstm(z)
        return torch.sigmoid(self.out_proj(self.norm(out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def filter_frames_by_temperature(frames: dict[str, list[pd.DataFrame]], temp_c: float) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    for split, split_frames in frames.items():
        for frame in split_frames:
            if len(frame) and np.isclose(float(frame["temperature"].iloc[0]), float(temp_c), atol=1e-6):
                out[split].append(frame)
    return out


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
    if loader is None:
        return float("nan")
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y, _meta in loader:
            pred = model(x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
            yy = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            errors.append(torch.abs(pred - yy).detach().cpu().numpy())
    if not errors:
        return float("nan")
    return float(np.mean(np.concatenate(errors)))


def train_one_temperature(
    cfg: NMCVcorrITSingleHeadByTempConfig,
    temp_c: float,
    frames: dict[str, list[pd.DataFrame]],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    temp_key = f"{temp_c:g}C".replace("-", "N")
    model_name = f"{cfg.output_prefix}_{temp_key}"
    temp_frames = filter_frames_by_temperature(frames, temp_c)
    if not temp_frames["train"] or not temp_frames["valid"] or not temp_frames["test"]:
        raise RuntimeError(
            f"Empty split for {temp_c:g}C: "
            f"train={len(temp_frames['train'])} valid={len(temp_frames['valid'])} test={len(temp_frames['test'])}"
        )
    scaled, _ = make_scaled_frames_for_ablation(temp_frames, FEATURE_COLS)

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
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Empty dataset for {temp_c:g}C: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}"
        )

    model = SingleHeadStatelessLSTM(input_dim=len(FEATURE_COLS), hidden_size=int(cfg.hidden_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)

    history = []
    best_valid_mae = float("inf")
    best_epoch = 0
    best_state = None
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        group_losses: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            if cfg.sequence_loss:
                pred = model.forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
            else:
                pred = model(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)

            drives = [str(v) for v in meta["drive_cycle"]]
            if cfg.rex_group == "drive":
                keys = [f"D{d}" for d in drives]
            else:
                temps = [float(v) for v in meta["temperature"]]
                keys = [f"T{t:g}_{d}" for t, d in zip(temps, drives)]
            per_group = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                per_group.append(g_loss)
                group_losses.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(per_group) if per_group else sample_loss.mean().view(1)
            mean_loss = stack.mean()
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(cfg.lambda_rex) * rex_var
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        row = {
            "model_name": model_name,
            "temperature_C": float(temp_c),
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
        }
        for key, vals in group_losses.items():
            row[f"train_loss_group_{key}"] = float(np.mean(vals))
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
    lookup = make_endpoint_lookup(temp_frames, FEATURE_COLS)
    pred = attach_eval_features(predict_loader(model, test_loader, model_name), lookup)
    valid_pred = attach_eval_features(predict_loader(model, valid_loader, model_name), lookup)
    for df in (pred, valid_pred):
        df["seed"] = int(cfg.seed)
        df["temperature_train_test_mode"] = "single_temperature"
        df["trained_temperature_C"] = float(temp_c)
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
            df["trained_temperature_C"] = float(temp_c)
            df["selected_epoch"] = int(best_epoch)
            df["input_feature_dim"] = len(FEATURE_COLS)

    overall.to_csv(out_dir / f"{model_name}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{model_name}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{model_name}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{model_name}_focus.csv", index=False)
    valid_overall.to_csv(out_dir / f"{model_name}_valid_overall.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{model_name}_valid_by_temperature.csv", index=False)

    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "model_name": model_name,
        "trained_temperature_C": float(temp_c),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "input_dim_explanation": "50 timesteps x 3 features: V_corr_raw, I_raw, T",
        "lstm_layers_verified": 1,
        "hidden_size_verified": int(cfg.hidden_size),
        "window_len_verified": int(cfg.window_len),
        "out_proj": "Linear(64 -> 1), single layer",
        "head_has_hidden_mlp": False,
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_amp": False,
        "label_column": "SOC_CC",
        "selected_epoch": int(best_epoch),
        "best_valid_MAE": float(best_valid_mae),
        "train_windows": int(len(train_ds)),
        "valid_windows": int(len(valid_ds)),
        "test_windows": int(len(test_ds)),
        "train_trajectories": int(len(temp_frames["train"])),
        "valid_trajectories": int(len(temp_frames["valid"])),
        "test_trajectories": int(len(temp_frames["test"])),
    }
    (out_dir / f"{model_name}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Overall:")
    print(overall.to_string(index=False), flush=True)
    print("Valid overall:")
    print(valid_overall.to_string(index=False), flush=True)
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


def write_report(
    cfg: NMCVcorrITSingleHeadByTempConfig,
    out_dir: Path,
    start_audit: pd.DataFrame,
    r0_df: pd.DataFrame,
    schema: pd.DataFrame,
    leakage: pd.DataFrame,
    overall: pd.DataFrame,
    valid_overall: pd.DataFrame,
    by_temp: pd.DataFrame,
    focus: pd.DataFrame,
) -> None:
    lines = [
        "# NMC Vcorr/I/T Single-Head LSTM By-Temperature",
        "",
        "## Fixed condition",
        f"- Raw data: `{cfg.raw_root}`",
        f"- Train profiles: {', '.join(cfg.train_profiles)}",
        f"- Valid profiles: {', '.join(cfg.valid_profiles)}",
        f"- Test profiles: {', '.join(cfg.test_profiles)}",
        "- Temperature mode: one model per temperature; train/valid/test never mix temperatures.",
        f"- Model: input projection 3->64, LSTM layer=1 hidden=64, LayerNorm, out_proj Linear(64->1), sigmoid",
        f"- Window={cfg.window_len}, train stride={cfg.stride}, valid/test stride=1",
        f"- Loss: sequence Huber beta={cfg.huber_beta} + REx {cfg.lambda_rex} grouped by `{cfg.rex_group}`",
        "- Inputs: V_corr_raw, I_raw, T only.",
        "- No SOC input, no cumulative Ah, no absolute time/progress, no window-local timestep, no explicit SOC current-integration update.",
        "",
        "## Test overall",
        table_md(overall, ["trained_temperature_C", "model_name", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Valid overall",
        table_md(valid_overall, ["trained_temperature_C", "model_name", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Test by temperature",
        table_md(by_temp, ["trained_temperature_C", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio"]),
        "",
        "## Focus",
        table_md(focus, ["trained_temperature_C", "scope", "n_windows", "MAE_pct", "RMSE_pct", "catastrophic_gt5_pct"]),
        "",
        "## R0 / Vcorr preprocessing",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
        "",
        "## Input schema",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## Leakage audit",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## Start SOC audit",
        table_md(
            start_audit,
            [
                "file_name",
                "temperature_C",
                "profile",
                "soc0_used",
                "soc0_vinit_v",
                "qnet_denom_Ah",
                "first_soc_cc",
            ],
        ),
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: NMCVcorrITSingleHeadByTempConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.layers) != 1 or int(cfg.window_len) != 50:
        raise ValueError("This experiment is fixed to hidden_size=64, layers=1, window_len=50.")

    out_dir = cfg.base_dir / "nmc_vcorr_it_lstm_singlehead_bytemp_results"
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
    for temp in cfg.temperatures:
        set_seed(cfg.seed)
        results.append(train_one_temperature(cfg, float(temp), frames, out_dir))

    overall = pd.concat([r["overall"] for r in results], ignore_index=True)
    valid_overall = pd.concat([r["valid_overall"] for r in results], ignore_index=True)
    by_temp = pd.concat([r["by_temperature"] for r in results], ignore_index=True)
    by_traj = pd.concat([r["by_trajectory"] for r in results], ignore_index=True)
    focus = pd.concat([r["focus"] for r in results], ignore_index=True)
    history = pd.concat([r["history"] for r in results], ignore_index=True)
    pred = pd.concat([r["pred"] for r in results], ignore_index=True)
    valid_pred = pd.concat([r["valid_pred"] for r in results], ignore_index=True)

    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    valid_overall.to_csv(out_dir / f"{cfg.output_prefix}_valid_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{cfg.output_prefix}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    pred.to_csv(out_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    valid_pred.to_csv(out_dir / f"{cfg.output_prefix}_valid_prediction_rows.csv.gz", index=False, compression="gzip")
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "lstm_layers_verified": 1,
        "hidden_size_verified": 64,
        "window_len_verified": 50,
        "out_proj": "Linear(64 -> 1), single layer",
        "temperature_mode": "one model per temperature",
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, valid_overall, by_temp, focus)
    print("By-temperature single-head test overall:")
    print(overall.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    return {
        "overall": overall,
        "valid_overall": valid_overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "history": history,
        "prediction_rows": pred,
        "valid_prediction_rows": valid_pred,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC Vcorr/I/T single-head 1-layer LSTM per-temperature experiment.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=NMCVcorrITSingleHeadByTempConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperatures", default="0,25,45")
    p.add_argument("--train-profiles", default="DST,US06")
    p.add_argument("--valid-profiles", default="NONE")
    p.add_argument("--test-profiles", default="FUDS")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--valid-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NMCVcorrITSingleHeadByTempConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        temperatures=tuple(float(s.strip()) for s in str(args.temperatures).split(",") if s.strip()),
        train_profiles=tuple(s.strip() for s in str(args.train_profiles).split(",") if s.strip()),
        valid_profiles=tuple(s.strip() for s in str(args.valid_profiles).split(",") if s.strip()),
        test_profiles=tuple(s.strip() for s in str(args.test_profiles).split(",") if s.strip()),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        valid_every=int(args.valid_every),
    )
    run(cfg)


if __name__ == "__main__":
    main()
