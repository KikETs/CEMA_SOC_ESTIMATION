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
from .nmc_vcorr_it_designed_rnn_experiment import DesignedRNNModel, DesignedVariant
from .nmc_vcorr_it_goal_remote_screen import VcorrITGoalModel
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


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_fixed_specialists_seed0"


@dataclass
class FixedGoalSpecialistConfig:
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
    epochs: int = 120
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    lambda_rex: float = 2.0
    rex_group: str = "drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 20
    valid_every: int = 5
    variant_names: tuple[str, ...] = ()
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class SpecialistVariant:
    name: str
    model_family: str = "goal"
    recurrent: str = "lstm"
    layers: int = 2
    head_kind: str = "mlp"
    temp_mode: str = "none"
    dropout: float = 0.05
    kernel_size: int = 5
    norm_kind: str = "channel"
    sequence_loss: bool = True
    lambda_rex: float = 2.0
    lr: float = 8e-4
    correction_limit: float = 0.8
    use_mha: bool = True


DESIGN_INTENT = [
    {
        "component": "temperature specialist",
        "reason": (
            "Existing fixed-input runs showed opposite FUDS bias patterns between 0C and 25C. "
            "A single shared encoder can average those incompatible offsets, so this screen trains one "
            "model per temperature while still selecting the model by the observed ambient temperature only."
        ),
    },
    {
        "component": "Vcorr/I/T-only window",
        "reason": (
            "The fixed goal keeps input columns to V_corr_raw, I_raw, and T for 50 causal samples. "
            "No SOC, cumulative Ah, absolute time, or explicit current-integration state is available."
        ),
    },
    {
        "component": "sequence TCN/LSTM variants",
        "reason": (
            "Prior checkpoint sweeps found the best 0C behavior in short-history temporal models, so the "
            "screen keeps causal LSTM/TCN encoders and uses sequence Huber loss for dense supervision."
        ),
    },
    {
        "component": "anchor plus attention RNN variant",
        "reason": (
            "FUDS has weakly observable plateau regions and transient current-change regions in the same "
            "50 s window. The anchor branch uses endpoint V_corr/T, while attention lets the recurrent "
            "branch focus on informative relaxation or excitation segments. The dynamic correction is bounded."
        ),
    },
]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def specialist_variants() -> list[SpecialistVariant]:
    return [
        SpecialistVariant(
            name="temp_lstm2_mlp_seq_rex2",
            model_family="goal",
            recurrent="lstm",
            layers=2,
            head_kind="mlp",
            sequence_loss=True,
            lambda_rex=2.0,
        ),
        SpecialistVariant(
            name="temp_lstm2_linear_seq_rex2",
            model_family="goal",
            recurrent="lstm",
            layers=2,
            head_kind="linear",
            sequence_loss=True,
            lambda_rex=2.0,
        ),
        SpecialistVariant(
            name="temp_tcn5_linear_seq_rex2",
            model_family="goal",
            recurrent="tcn",
            layers=5,
            head_kind="linear",
            dropout=0.04,
            sequence_loss=True,
            lambda_rex=2.0,
        ),
        SpecialistVariant(
            name="temp_anchor_lstm_mha_lim0p8_rex2",
            model_family="designed",
            recurrent="lstm",
            layers=1,
            dropout=0.04,
            sequence_loss=False,
            lambda_rex=2.0,
            correction_limit=0.8,
            use_mha=True,
        ),
        SpecialistVariant(
            name="temp_anchor_gru_mha_lim0p8_rex2",
            model_family="designed",
            recurrent="gru",
            layers=1,
            dropout=0.04,
            sequence_loss=False,
            lambda_rex=2.0,
            correction_limit=0.8,
            use_mha=True,
        ),
    ]


def filter_frames_by_temperature(frames: dict[str, list[pd.DataFrame]], temp_c: float) -> dict[str, list[pd.DataFrame]]:
    out = {"train": [], "valid": [], "test": []}
    for split, split_frames in frames.items():
        for frame in split_frames:
            if len(frame) and np.isclose(float(frame["temperature"].iloc[0]), float(temp_c), atol=1e-6):
                out[split].append(frame)
    return out


def build_model(variant: SpecialistVariant, hidden_size: int) -> nn.Module:
    if variant.model_family == "designed":
        designed = DesignedVariant(
            name=variant.name,
            encoder=variant.recurrent,
            layers=int(variant.layers),
            temp_mode="none",
            use_anchor=True,
            use_mha=bool(variant.use_mha),
            correction_limit=float(variant.correction_limit),
            dropout=float(variant.dropout),
            lambda_rex=float(variant.lambda_rex),
            lr=float(variant.lr),
        )
        return DesignedRNNModel(designed, input_dim=len(FEATURE_COLS), hidden_size=int(hidden_size))
    return VcorrITGoalModel(
        input_dim=len(FEATURE_COLS),
        hidden_size=int(hidden_size),
        recurrent=variant.recurrent,
        layers=int(variant.layers),
        head_kind=variant.head_kind,
        temp_mode=variant.temp_mode,
        dropout=float(variant.dropout),
        kernel_size=int(variant.kernel_size),
        norm_kind=variant.norm_kind,
    )


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


def group_keys(meta, group_name: str) -> list[str]:
    temps = [float(v) for v in meta["temperature"]]
    drives = [str(v) for v in meta["drive_cycle"]]
    if group_name == "temperature":
        return [f"T{t:g}" for t in temps]
    if group_name == "drive":
        return [f"D{d}" for d in drives]
    return [f"T{t:g}_{d}" for t, d in zip(temps, drives)]


def train_one(
    cfg: FixedGoalSpecialistConfig,
    variant: SpecialistVariant,
    temp_c: float,
    frames: dict[str, list[pd.DataFrame]],
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    temp_key = f"{temp_c:g}C".replace("-", "N")
    model_name = f"{cfg.output_prefix}_{variant.name}_{temp_key}"
    temp_frames = filter_frames_by_temperature(frames, temp_c)
    if not temp_frames["train"] or not temp_frames["valid"] or not temp_frames["test"]:
        raise RuntimeError(f"Empty split for {variant.name} {temp_c:g}C")
    scaled, _ = make_scaled_frames_for_ablation(temp_frames, FEATURE_COLS)

    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0

    train_cls = SequenceWindowDataset if bool(variant.sequence_loss) else DecomposedWindowDataset
    train_ds = train_cls(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    model = build_model(variant, cfg.hidden_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
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
        group_loss_log: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            if bool(variant.sequence_loss):
                pred = model.forward_sequence(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
            else:
                pred = model(x)
                sample_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            keys = group_keys(meta, cfg.rex_group)
            per_group = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                per_group.append(g_loss)
                group_loss_log.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(per_group) if per_group else sample_loss.mean().view(1)
            mean_loss = stack.mean()
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(variant.lambda_rex) * rex_var
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        row = {
            "model_name": model_name,
            "variant": variant.name,
            "trained_temperature_C": float(temp_c),
            "epoch": int(ep),
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
        }
        for key, vals in group_loss_log.items():
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
        df["variant"] = variant.name
        df["trained_temperature_C"] = float(temp_c)
        df["selected_epoch"] = int(best_epoch)
        df["input_feature_dim"] = len(FEATURE_COLS)
        df["fixed_goal_schema"] = "V_corr_raw,I_raw,T|w50|h64|no_soc|no_cc"
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
            df["trained_temperature_C"] = float(temp_c)
            df["selected_epoch"] = int(best_epoch)
            df["input_feature_dim"] = len(FEATURE_COLS)
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


def write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, by_temp, valid_overall, selected):
    lines = [
        "# Fixed Goal Vcorr/I/T Temperature Specialists",
        "",
        "## Fixed values",
        "- Inputs: V_corr_raw, I_raw, T only.",
        "- Window length: 50.",
        "- Hidden/model dimension: 64.",
        "- Train: DST + US06, valid: VALIDATION, test: FUDS.",
        "- No SOC input, no cumulative Ah, no absolute time/progress, no explicit current-integration SOC update.",
        "- Goal remains: FUDS 0C MAE < 1.0%p and 25C MAE < 0.7%p.",
        "",
        "## Design intent",
    ]
    lines += [f"- {row['component']}: {row['reason']}" for row in DESIGN_INTENT]
    lines += [
        "",
        "## Test overall",
        table_md(overall, ["variant", "trained_temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Test by temperature",
        table_md(by_temp, ["variant", "trained_temperature_C", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio", "selected_epoch"]),
        "",
        "## Valid overall",
        table_md(valid_overall, ["variant", "trained_temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch"]),
        "",
        "## Valid-selected temperature specialist",
        table_md(selected, ["temperature_C", "selected_variant", "valid_MAE_pct", "test_MAE_pct", "test_RMSE_pct", "selected_epoch", "goal_relevant"]),
        "",
        "## Input schema",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## Leakage audit",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## R0 / Vcorr preprocessing",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
        "",
        "## Start SOC audit",
        table_md(start_audit, ["file_name", "temperature_C", "profile", "soc0_used", "first_soc_cc"]),
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def valid_selected_table(valid_by_temp: pd.DataFrame, by_temp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temp in sorted(valid_by_temp["temperature_C"].dropna().unique()):
        valid_sub = valid_by_temp[np.isclose(valid_by_temp["temperature_C"], float(temp))].copy()
        if valid_sub.empty:
            continue
        best = valid_sub.sort_values("MAE_pct").iloc[0]
        test_sub = by_temp[
            np.isclose(by_temp["temperature_C"], float(temp))
            & (by_temp["variant"] == best["variant"])
            & np.isclose(by_temp["trained_temperature_C"], float(best["trained_temperature_C"]))
        ]
        test = test_sub.iloc[0] if len(test_sub) else None
        rows.append(
            {
                "temperature_C": float(temp),
                "selected_variant": str(best["variant"]),
                "valid_MAE_pct": float(best["MAE_pct"]),
                "test_MAE_pct": float(test["MAE_pct"]) if test is not None else float("nan"),
                "test_RMSE_pct": float(test["RMSE_pct"]) if test is not None else float("nan"),
                "selected_epoch": int(best["selected_epoch"]),
                "goal_relevant": bool(np.isclose(float(temp), 0.0) or np.isclose(float(temp), 25.0)),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: FixedGoalSpecialistConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("Fixed goal requires hidden_size=64 and window_len=50.")
    if FEATURE_COLS != ["V_corr_raw", "I_raw", "T"]:
        raise ValueError(f"Unexpected feature columns: {FEATURE_COLS}")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_fixed_specialist_results"
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

    variants = specialist_variants()
    if cfg.variant_names:
        selected_names = set(cfg.variant_names)
        variants = [v for v in variants if v.name in selected_names]
        missing = sorted(selected_names - {v.name for v in variants})
        if missing:
            raise ValueError(f"Unknown variant names: {missing}")
    results = []
    for variant in variants:
        for temp in cfg.temperatures:
            set_seed(cfg.seed)
            print(f"===== {variant.name} {temp:g}C =====", flush=True)
            results.append(train_one(cfg, variant, float(temp), frames, out_dir))

    overall = pd.concat([r["overall"] for r in results], ignore_index=True)
    by_temp = pd.concat([r["by_temperature"] for r in results], ignore_index=True)
    by_traj = pd.concat([r["by_trajectory"] for r in results], ignore_index=True)
    focus = pd.concat([r["focus"] for r in results], ignore_index=True)
    valid_overall = pd.concat([r["valid_overall"] for r in results], ignore_index=True)
    valid_by_temp = pd.concat([r["valid_by_temperature"] for r in results], ignore_index=True)
    history = pd.concat([r["history"] for r in results], ignore_index=True)
    pred = pd.concat([r["pred"] for r in results], ignore_index=True)
    valid_pred = pd.concat([r["valid_pred"] for r in results], ignore_index=True)
    selected = valid_selected_table(valid_by_temp, by_temp)
    goal_met = bool(
        len(selected)
        and (selected.loc[np.isclose(selected["temperature_C"], 0.0), "test_MAE_pct"].min() < 1.0)
        and (selected.loc[np.isclose(selected["temperature_C"], 25.0), "test_MAE_pct"].min() < 0.7)
    )
    for df in (overall, by_temp, by_traj, focus, valid_overall, valid_by_temp, selected):
        if not df.empty:
            df["goal_0C_lt1_25C_lt0p7"] = goal_met

    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{cfg.output_prefix}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    valid_overall.to_csv(out_dir / f"{cfg.output_prefix}_valid_overall.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{cfg.output_prefix}_valid_by_temperature.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    pred.to_csv(out_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    valid_pred.to_csv(out_dir / f"{cfg.output_prefix}_valid_prediction_rows.csv.gz", index=False, compression="gzip")
    selected.to_csv(out_dir / f"{cfg.output_prefix}_valid_selected_specialists.csv", index=False)

    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "hidden_size_verified": 64,
        "window_len_verified": 50,
        "goal_met": goal_met,
        "design_intent": DESIGN_INTENT,
        "variants": [asdict(v) for v in variants],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, start_audit, r0_df, schema, leakage, overall, by_temp, valid_overall, selected)
    print("Fixed specialist valid-selected:")
    print(selected.to_string(index=False), flush=True)
    print("Fixed specialist by temperature:")
    print(by_temp.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    return {
        "overall": overall,
        "by_temperature": by_temp,
        "by_trajectory": by_traj,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
        "history": history,
        "prediction_rows": pred,
        "valid_prediction_rows": valid_pred,
        "valid_selected": selected,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-goal Vcorr/I/T temperature specialist screen.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=FixedGoalSpecialistConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperatures", default="0,25,45")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--valid-every", type=int, default=5)
    p.add_argument("--variant-names", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FixedGoalSpecialistConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        temperatures=tuple(float(s.strip()) for s in str(args.temperatures).split(",") if s.strip()),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        valid_every=int(args.valid_every),
        variant_names=tuple(s.strip() for s in str(args.variant_names).split(",") if s.strip()),
    )
    run(cfg)


if __name__ == "__main__":
    main()
