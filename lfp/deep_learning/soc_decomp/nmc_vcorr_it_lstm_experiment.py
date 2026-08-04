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
from .deep_no_leak_experiment import DeepNoLeakLSTM, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset
from .nmc_branchbands_experiment import (
    FORBIDDEN_INPUT_PATTERNS,
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
    focus_metrics,
    metrics_by_trajectory,
    table_md,
    write_start_audit,
)
from .runtime import configure_torch_runtime, device
from .training import attach_prediction_features, build_prediction_feature_lookup, make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


FEATURE_COLS = ["V_corr_raw", "I_raw", "T"]


@dataclass
class NMCVcorrITLSTMConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = "nmc_vcorr_I_T_lstm_w50_alltemps_trainProfiles_to_FUDS_seed0"
    seed: int = 0
    train_profiles: tuple[str, ...] = ("VALIDATION", "DST", "US06")
    valid_profiles: tuple[str, ...] = ()
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.05
    lambda_rex: float = 2.0
    rex_group: str = "temperature_drive"
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 10
    valid_every: int = 10
    low_current_threshold_A: float = 0.05
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


def write_input_schema(out_path: Path) -> pd.DataFrame:
    rows = [
        {
            "index_1based": 1,
            "feature_name": "V_corr_raw",
            "source": "causal ohmic-corrected voltage EMA proxy",
            "uses_soc_input": False,
            "uses_cumulative_input": False,
            "uses_explicit_current_integration": False,
        },
        {
            "index_1based": 2,
            "feature_name": "I_raw",
            "source": "instantaneous current excitation only",
            "uses_soc_input": False,
            "uses_cumulative_input": False,
            "uses_explicit_current_integration": False,
        },
        {
            "index_1based": 3,
            "feature_name": "T",
            "source": "ambient temperature label",
            "uses_soc_input": False,
            "uses_cumulative_input": False,
            "uses_explicit_current_integration": False,
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_leakage_audit(source_columns: list[str], out_path: Path) -> pd.DataFrame:
    selected_bad = [c for c in FEATURE_COLS if any(tok in c for tok in FORBIDDEN_INPUT_PATTERNS)]
    source_forbidden = [c for c in source_columns if any(tok in c for tok in FORBIDDEN_INPUT_PATTERNS)]
    rows = [
        {
            "audit_item": "selected_input_columns_forbidden_name_scan",
            "status": "PASS" if not selected_bad else "FAIL",
            "detail": ",".join(selected_bad) if selected_bad else "Selected inputs are exactly V_corr_raw, I_raw, T.",
        },
        {
            "audit_item": "source_has_forbidden_columns_but_not_selected",
            "status": "PASS",
            "detail": ",".join(source_forbidden),
        },
        {
            "audit_item": "explicit_soc_state_update",
            "status": "PASS",
            "detail": "No SOC_{t+1}=SOC_t-I*dt/Q update is used; SOC_CC is label only.",
        },
        {
            "audit_item": "cumulative_current_features",
            "status": "PASS",
            "detail": "No SOC_CC, cumulative Ah, progress, capacity, absolute time, or window-local time feature is selected.",
        },
        {
            "audit_item": "current_usage",
            "status": "PASS",
            "detail": "Current is used as instantaneous I_raw only; it is not integrated into an SOC state.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    if selected_bad:
        raise RuntimeError(f"CUMULATIVE_FEATURE_LEAK: forbidden selected input columns: {selected_bad}")
    return out


def predict_loader(model: torch.nn.Module, loader, model_name: str) -> pd.DataFrame:
    from .models import collate_meta_to_frame

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


def evaluate_loader_mae(model: torch.nn.Module, loader) -> float:
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


def add_minimal_extra_features(pred: pd.DataFrame, frames: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    if pred.empty:
        return pred
    rows = []
    keep = ["trajectory_id", "end_index", "V_corr_raw", "I_raw", "T", "absI"]
    for split_frames in frames.values():
        for frame in split_frames:
            have = [c for c in keep if c in frame.columns]
            rows.append(frame[have])
    lookup = pd.concat(rows, ignore_index=True).drop_duplicates(["trajectory_id", "end_index"])
    overlap = [c for c in lookup.columns if c not in {"trajectory_id", "end_index"} and c in pred.columns]
    lookup = lookup.drop(columns=overlap)
    return pred.merge(lookup, on=["trajectory_id", "end_index"], how="left", validate="many_to_one")


def write_report(
    cfg: NMCVcorrITLSTMConfig,
    out_dir: Path,
    start_audit: pd.DataFrame,
    r0_df: pd.DataFrame,
    schema: pd.DataFrame,
    leakage: pd.DataFrame,
    overall: pd.DataFrame,
    by_temp: pd.DataFrame,
    by_traj: pd.DataFrame,
    focus: pd.DataFrame,
) -> None:
    lines = [
        "# NMC Vcorr-I-T Stateless LSTM 결과",
        "",
        "## 설정",
        f"- Raw data: `{cfg.raw_root}`",
        f"- Train profiles: {', '.join(cfg.train_profiles)}",
        f"- Valid profiles: {', '.join(cfg.valid_profiles) if cfg.valid_profiles else '(none)'}",
        f"- Test profiles: {', '.join(cfg.test_profiles)}",
        "- Train temperatures: 0, 25, 45 C all included",
        f"- Model: stateless LSTM, window={cfg.window_len}, stride={cfg.stride}, hidden={cfg.hidden_size}, layers={cfg.layers}",
        f"- Inputs: {', '.join(FEATURE_COLS)}",
        "- Actual input tensor: 50 timesteps x 3 features; no delta-start feature and no timestep feature",
        f"- Objective: endpoint Huber beta={cfg.huber_beta}, REx={cfg.lambda_rex} grouped by `{cfg.rex_group}`",
        "- Strict NoCC: SOC input 없음, cumulative Ah/progress/time 입력 없음, explicit SOC current-integration state update 없음",
        "- Current usage: instantaneous I_raw only, not integrated into SOC state",
        "",
        "## SOC 80% 시작점 감사",
        table_md(
            start_audit,
            [
                "file_name",
                "temperature_C",
                "profile",
                "first_data_point",
                "first_test_time_s",
                "first_step_index",
                "first_soc_cc",
                "qnet_denom_Ah",
                "starts_at_80pct",
            ],
        ),
        "",
        "## V_corr 생성",
        "- V_corr_raw는 이전 NMC BranchBands 실험과 같은 causal proxy decomposition으로 생성했다.",
        "- R0는 test profile(FUDS)을 제외하고 train profiles에서 온도별 전류 step의 robust dV/dI median으로 추정했다.",
        "- 이후 `V_ohm_raw = I_raw * R0`, `V_corr_raw = causal EMA(V_raw - V_ohm_raw)`로 계산했다.",
        "",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
        "",
        "## 입력 스키마",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## 누수 감사",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## 성능",
        "SOC 값은 0-1 scale에서 학습했고 아래 MAE/RMSE는 %-point로 표시했다.",
        "",
        "### Overall",
        table_md(overall, ["model_name", "n_windows", "MAE_pct", "RMSE_pct", "error_std_pct"]),
        "",
        "### By Temperature",
        table_md(by_temp, ["temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio"]),
        "",
        "### By Trajectory",
        table_md(by_traj, ["temperature_C", "drive_cycle", "trajectory_id", "n_windows", "MAE_pct", "RMSE_pct", "catastrophic_gt5_pct"]),
        "",
        "### Focus",
        table_md(focus, ["scope", "n_windows", "MAE_pct", "RMSE_pct", "catastrophic_gt5_pct"]),
        "",
        "## 해석 주의",
        "- 이 결과는 온도 0/25/45 C를 모두 포함하되 profile split을 분리한 결과다.",
        "- valid profile이 지정된 경우 valid MAE가 가장 낮은 checkpoint로 test를 평가했다.",
        "- current는 사용하지 않은 것이 아니라 instantaneous excitation으로만 사용했다.",
        "- NoCC ablation 결과이지 current integration이 불필요하다는 증명은 아니다.",
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: NMCVcorrITLSTMConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    out_dir = cfg.base_dir / "nmc_vcorr_it_lstm_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    set_seed(cfg.seed)

    files = find_csv_files(cfg.raw_root)
    start_audit = write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = build_feature_frames(cfg, files, r0_df)
    schema = write_input_schema(out_dir / f"{cfg.output_prefix}_input_schema.csv")
    leakage = write_leakage_audit(raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")

    scaled, _ = make_scaled_frames_for_ablation(frames, FEATURE_COLS)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0

    train_ds = DecomposedWindowDataset(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = (
        DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
        if len(scaled.get("valid", []))
        else None
    )
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Empty dataset: train_windows={len(train_ds)} test_windows={len(test_ds)}")

    model_name = cfg.output_prefix
    model = DeepNoLeakLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_size=cfg.hidden_size,
        layers=cfg.layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg) if valid_ds is not None and len(valid_ds) else None
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
            pred = model(x)
            sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            temps = [float(v) for v in meta["temperature"]]
            drives = [str(v) for v in meta["drive_cycle"]]
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
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
        }
        for key, vals in group_losses.items():
            safe = str(key).replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))
        if valid_loader is not None and (
            ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.valid_every)) == 0
        ):
            valid_mae = evaluate_loader_mae(model, valid_loader)
            row["valid_MAE"] = float(valid_mae)
            row["valid_MAE_pct"] = float(valid_mae * 100.0)
            if np.isfinite(valid_mae) and valid_mae < best_valid_mae:
                best_valid_mae = float(valid_mae)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            valid_msg = ""
            if "valid_MAE_pct" in row:
                valid_msg = f" valid_mae={row['valid_MAE_pct']:.3f}%"
            print(f"{model_name} epoch={ep} loss={row['loss']:.5f}{valid_msg}", flush=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        best_epoch = int(cfg.epochs)
    pred = predict_loader(model, test_loader, model_name)
    valid_pred = predict_loader(model, valid_loader, model_name) if valid_loader is not None else pd.DataFrame()
    generic_lookup = build_prediction_feature_lookup(frames)
    pred = attach_prediction_features(
        pred.assign(split="test", ablation=model_name),
        generic_lookup,
        ablation_name=model_name,
        target_label="physical",
    )
    pred = add_minimal_extra_features(pred, frames)
    pred["seed"] = int(cfg.seed)
    pred["train_profiles"] = ",".join(cfg.train_profiles)
    pred["valid_profiles"] = ",".join(cfg.valid_profiles)
    pred["test_profiles"] = ",".join(cfg.test_profiles)
    pred["selected_epoch"] = int(best_epoch)
    pred.to_csv(out_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    if not valid_pred.empty:
        valid_pred = attach_prediction_features(
            valid_pred.assign(split="valid", ablation=model_name),
            generic_lookup,
            ablation_name=model_name,
            target_label="physical",
        )
        valid_pred = add_minimal_extra_features(valid_pred, frames)
        valid_pred["seed"] = int(cfg.seed)
        valid_pred["train_profiles"] = ",".join(cfg.train_profiles)
        valid_pred["valid_profiles"] = ",".join(cfg.valid_profiles)
        valid_pred["test_profiles"] = ",".join(cfg.test_profiles)
        valid_pred["selected_epoch"] = int(best_epoch)
        valid_pred.to_csv(out_dir / f"{cfg.output_prefix}_valid_prediction_rows.csv.gz", index=False, compression="gzip")

    overall = _overall_metrics(pred)
    by_temp = variance_by_temperature(pred)
    by_traj = metrics_by_trajectory(pred)
    focus = focus_metrics(pred, cfg, model_name)
    valid_overall = _overall_metrics(valid_pred) if not valid_pred.empty else pd.DataFrame()
    valid_by_temp = variance_by_temperature(valid_pred) if not valid_pred.empty else pd.DataFrame()
    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{cfg.output_prefix}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    if not valid_overall.empty:
        valid_overall.to_csv(out_dir / f"{cfg.output_prefix}_valid_overall.csv", index=False)
        valid_by_temp.to_csv(out_dir / f"{cfg.output_prefix}_valid_by_temperature.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "model_name": model_name,
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "input_dim_explanation": "50 timesteps x 3 features: V_corr_raw, I_raw, T",
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_self_supervised_pretraining": False,
        "label_column": "SOC_CC",
        "selected_epoch": int(best_epoch),
        "best_valid_MAE": float(best_valid_mae) if np.isfinite(best_valid_mae) else None,
        "train_windows": int(len(train_ds)),
        "valid_windows": int(len(valid_ds)) if valid_ds is not None else 0,
        "test_windows": int(len(test_ds)),
        "train_trajectories": int(len(frames["train"])),
        "valid_trajectories": int(len(frames["valid"])),
        "test_trajectories": int(len(frames["test"])),
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, by_temp, by_traj, focus)
    print("Overall:")
    print(overall.to_string(index=False), flush=True)
    if not valid_overall.empty:
        print("Valid overall:")
        print(valid_overall.to_string(index=False), flush=True)
        print("Selected epoch:")
        print(best_epoch, flush=True)
    print("By temperature:")
    print(by_temp.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    return {
        "history": history_df,
        "pred": pred,
        "overall": overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC Vcorr/I/T-only stateless LSTM experiment.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default="nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    p.add_argument("--output-prefix", default=NMCVcorrITLSTMConfig.output_prefix)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-profiles", default="VALIDATION,DST,US06")
    p.add_argument("--valid-profiles", default="")
    p.add_argument("--test-profiles", default="FUDS")
    p.add_argument("--window-len", type=int, default=50)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--valid-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NMCVcorrITLSTMConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        train_profiles=tuple(s.strip() for s in str(args.train_profiles).split(",") if s.strip()),
        valid_profiles=tuple(s.strip() for s in str(args.valid_profiles).split(",") if s.strip()),
        test_profiles=tuple(s.strip() for s in str(args.test_profiles).split(",") if s.strip()),
        window_len=int(args.window_len),
        stride=int(args.stride),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        dropout=float(args.dropout),
        lambda_rex=float(args.lambda_rex),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        valid_every=int(args.valid_every),
    )
    run(cfg)


if __name__ == "__main__":
    main()
