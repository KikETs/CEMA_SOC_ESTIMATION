#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_decomp.nmc_vcorr_it_goal_remote_screen import VcorrITGoalModel, set_seed
from soc_decomp.nmc_vcorr_it_train_dst_selector_run import (
    TrainDSTSelectorConfig,
    _cached_feature_frames,
    _cached_find_csv_files,
    _cached_r0_by_temperature,
    _cached_raw_source_columns,
    _cached_scaled_frames_for_ablation,
    _filter_scaled_frames_by_temperatures,
    _selected_feature_columns,
    _split_train_profile_blocks_for_validation,
)
from soc_decomp.nmc_branchbands_experiment import write_start_audit
from soc_decomp.nmc_vit_feature_lstm_experiment import write_input_schema, write_leakage_audit
from soc_decomp.runtime import configure_torch_runtime, device


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")


@dataclass(frozen=True)
class StatefulConfig:
    base_dir: Path
    raw_root: Path
    holdout: str = "VALIDATION"
    feature_set: str = "paper_g4_all_ema"
    output_prefix: str = "soc80_goal_w50_stateful_gru_stream"
    seeds: tuple[int, ...] = (1,)
    epochs: int = 200
    chunk_len: int = 50
    hidden_size: int = 128
    layers: int = 1
    dropout: float = 0.06
    lr: float = 8e-4
    weight_decay: float = 2e-4
    huber_beta: float = 0.02
    weight_0: float = 1.2
    weight_25: float = 2.2
    weight_45: float = 1.0
    lambda_rex: float = 1.0
    lambda_cold_lowsoc_underpred_loss: float = 1.0
    cold_lowsoc_underpred_soc_upper: float = 0.5
    hard_region_loss_weight: float = 0.5
    hard_region_soc_lower: float = 0.0
    hard_region_soc_upper: float = 0.5
    grad_clip: float = 1.0
    print_every: int = 10
    save_predictions: bool = True


@dataclass
class FrameTensor:
    x: torch.Tensor
    y: torch.Tensor
    temperature: float
    drive_cycle: str
    file_name: str
    trajectory_id: str
    end_index: np.ndarray


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    holdout = str(holdout).upper()
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def _base_selector_cfg(cfg: StatefulConfig) -> TrainDSTSelectorConfig:
    return TrainDSTSelectorConfig(
        base_dir=Path(cfg.base_dir).resolve(),
        raw_root=Path(cfg.raw_root),
        output_prefix=str(cfg.output_prefix),
        seeds=tuple(int(s) for s in cfg.seeds),
        train_profiles=train_profiles_for_holdout(cfg.holdout),
        valid_profiles=("NONE",),
        test_profiles=(str(cfg.holdout).upper(),),
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0, 25.0, 45.0),
        window_len=int(cfg.chunk_len),
        stride=3,
        feature_set=str(cfg.feature_set),
        stage2_feature_set=str(cfg.feature_set),
        skip_stage2=True,
        valid_split_mode="profile",
    )


def _prepare_frames(cfg: StatefulConfig, out_dir: Path) -> tuple[list[str], dict[str, list[pd.DataFrame]]]:
    selector_cfg = _base_selector_cfg(cfg)
    selector_cfg.base_dir = Path(selector_cfg.base_dir).resolve()
    if not Path(selector_cfg.raw_root).is_absolute():
        selector_cfg.raw_root = selector_cfg.base_dir / selector_cfg.raw_root
    files = _cached_find_csv_files(selector_cfg.raw_root)
    raw_source_columns = _cached_raw_source_columns(files[0])
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = _cached_r0_by_temperature(selector_cfg, files)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = _cached_feature_frames(selector_cfg, files, r0_df)
    frames = _split_train_profile_blocks_for_validation(
        frames,
        selector_cfg,
        out_dir / f"{cfg.output_prefix}_internal_valid_split_audit.csv",
    )
    feature_cols = _selected_feature_columns(str(cfg.feature_set))
    available = set().union(*(set(frame.columns) for split_frames in frames.values() for frame in split_frames))
    missing = [col for col in feature_cols if col not in available]
    if missing:
        raise RuntimeError(f"Selected feature_set={cfg.feature_set!r} has missing columns: {missing}")
    write_input_schema(feature_cols, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    write_leakage_audit(feature_cols, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    scaled, _ = _cached_scaled_frames_for_ablation(frames, feature_cols)
    scaled = _filter_scaled_frames_by_temperatures(
        scaled,
        train_temperatures=(0.0, 25.0, 45.0),
        valid_temperatures=(),
        test_temperatures=(0.0, 25.0, 45.0),
        name="stateful",
    )
    return feature_cols, scaled


def _as_frame_tensors(frames: list[pd.DataFrame], feature_cols: list[str], chunk_len: int) -> list[FrameTensor]:
    out: list[FrameTensor] = []
    for frame in frames:
        if len(frame) < int(chunk_len):
            continue
        x_np = np.ascontiguousarray(frame[feature_cols].to_numpy(np.float32)).copy()
        y_np = np.ascontiguousarray(frame["SOC_physical"].to_numpy(np.float32)[:, None]).copy()
        if not np.isfinite(x_np).all() or not np.isfinite(y_np).all():
            raise RuntimeError(f"Non-finite stateful frame encountered: {frame['file_name'].iloc[0]}")
        out.append(
            FrameTensor(
                x=torch.as_tensor(x_np, device=device, dtype=torch.float32),
                y=torch.as_tensor(y_np, device=device, dtype=torch.float32),
                temperature=float(frame["temperature"].iloc[0]),
                drive_cycle=str(frame["drive_cycle"].iloc[0]).upper(),
                file_name=str(frame["file_name"].iloc[0]),
                trajectory_id=str(frame["trajectory_id"].iloc[0]),
                end_index=frame["end_index"].to_numpy(np.int64),
            )
        )
    return out


def _zero_state(model: VcorrITGoalModel) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    layers = int(model.rnn.num_layers)
    hidden = int(model.rnn.hidden_size)
    z = torch.zeros((layers, hidden), device=device, dtype=torch.float32)
    if model.recurrent == "lstm":
        return z, z.clone()
    return z


def _stack_states(
    model: VcorrITGoalModel,
    states: dict[int, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    frame_ids: list[int],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if model.recurrent not in {"lstm", "gru", "rnn"}:
        return None
    filled = [states.get(fid, _zero_state(model)) for fid in frame_ids]
    if model.recurrent == "lstm":
        h = torch.stack([item[0] for item in filled], dim=1)
        c = torch.stack([item[1] for item in filled], dim=1)
        return h, c
    return torch.stack(filled, dim=1)


def _split_state(
    model: VcorrITGoalModel,
    state_next: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    frame_ids: list[int],
    states: dict[int, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if state_next is None:
        return
    if model.recurrent == "lstm":
        h, c = state_next
        for j, fid in enumerate(frame_ids):
            states[fid] = (h[:, j, :].detach(), c[:, j, :].detach())
    else:
        for j, fid in enumerate(frame_ids):
            states[fid] = state_next[:, j, :].detach()


def _logits_from_hidden(model: VcorrITGoalModel, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    temp = x[..., model.temp_idx:model.temp_idx + 1]
    if model.temp_mode == "none":
        return model.base_head(h)
    if model.temp_mode == "bias":
        return model.base_head(h) + model.temp_bias(temp)
    if model.temp_mode == "hard_affine":
        base = model.base_head(h)
        idx = model._hard_head_indices(temp)
        flat_idx = idx.reshape(-1)
        scale = torch.exp(model.temp_affine_log_scale.clamp(-1.0, 1.0)).to(base.device, base.dtype)
        bias = model.temp_affine_bias.to(base.device, base.dtype)
        selected_scale = scale.index_select(0, flat_idx).reshape_as(base)
        selected_bias = bias.index_select(0, flat_idx).reshape_as(base)
        return selected_scale * base + selected_bias
    if model.temp_mode == "moe":
        logits = torch.stack([head(h) for head in model.expert_heads], dim=-1).squeeze(-2)
        gate = torch.softmax(model.temp_gate(temp), dim=-1)
        return torch.sum(logits * gate, dim=-1, keepdim=True)
    if model.temp_mode == "hard_heads":
        logits = torch.stack([head(h) for head in model.expert_heads], dim=-1).squeeze(-2)
        idx = model._hard_head_indices(temp)
        return torch.gather(logits, dim=-1, index=idx.unsqueeze(-1))
    raise ValueError(f"Unsupported temp_mode={model.temp_mode!r}")


def _stateful_forward(
    model: VcorrITGoalModel,
    x: torch.Tensor,
    state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None]:
    if model.recurrent not in {"lstm", "gru", "rnn"}:
        h = model.encode_sequence(x)
        return torch.sigmoid(_logits_from_hidden(model, h, x)), h, None
    z = model.input_proj(x)
    out, state_next = model.rnn(z, state)
    h = model.norm(out)
    return torch.sigmoid(_logits_from_hidden(model, h, x)), h, state_next


def _temp_weight(temp: float, cfg: StatefulConfig) -> float:
    if abs(float(temp) - 0.0) < 1e-6:
        return float(cfg.weight_0)
    if abs(float(temp) - 25.0) < 1e-6:
        return float(cfg.weight_25)
    if abs(float(temp) - 45.0) < 1e-6:
        return float(cfg.weight_45)
    return 1.0


def _train_one_seed(
    cfg: StatefulConfig,
    seed: int,
    feature_cols: list[str],
    train_frames: list[FrameTensor],
    out_dir: Path,
) -> tuple[VcorrITGoalModel, pd.DataFrame]:
    set_seed(int(seed))
    model = VcorrITGoalModel(
        input_dim=len(feature_cols),
        hidden_size=int(cfg.hidden_size),
        recurrent="gru",
        layers=int(cfg.layers),
        head_kind="gated_mlp",
        temp_mode="none",
        dropout=float(cfg.dropout),
        kernel_size=5,
        norm_kind="channel",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    max_chunks = max(int(frame.x.shape[0]) // int(cfg.chunk_len) for frame in train_frames)
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        states: dict[int, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
        losses = []
        for chunk_id in range(max_chunks):
            start = int(chunk_id) * int(cfg.chunk_len)
            end = start + int(cfg.chunk_len)
            active = [idx for idx, frame in enumerate(train_frames) if int(frame.x.shape[0]) >= end]
            if not active:
                continue
            x = torch.stack([train_frames[idx].x[start:end] for idx in active], dim=0)
            y = torch.stack([train_frames[idx].y[start:end] for idx in active], dim=0)
            state = _stack_states(model, states, active)
            opt.zero_grad(set_to_none=True)
            pred, _h, state_next = _stateful_forward(model, x, state)
            point_loss = F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none").squeeze(-1)
            valid_mask = torch.ones_like(point_loss, dtype=torch.bool)
            if start == 0 and int(cfg.chunk_len) > 1:
                valid_mask[:, : int(cfg.chunk_len) - 1] = False
            temp_w = torch.as_tensor(
                [_temp_weight(train_frames[idx].temperature, cfg) for idx in active],
                device=device,
                dtype=torch.float32,
            ).view(-1, 1)
            hard_mask = torch.zeros_like(point_loss, dtype=torch.bool)
            for row, idx in enumerate(active):
                if abs(train_frames[idx].temperature - 0.0) < 1e-6:
                    hard_mask[row] = (
                        y[row, :, 0].ge(float(cfg.hard_region_soc_lower))
                        & y[row, :, 0].le(float(cfg.hard_region_soc_upper))
                    )
            hard_weight = 1.0 + float(cfg.hard_region_loss_weight) * hard_mask.to(dtype=point_loss.dtype)
            weighted = point_loss * temp_w * hard_weight
            denom = (valid_mask.to(dtype=point_loss.dtype) * temp_w).sum().clamp_min(1.0)
            mean_loss = (weighted * valid_mask.to(dtype=point_loss.dtype)).sum() / denom
            group_losses = []
            for row in range(len(active)):
                mask = valid_mask[row]
                if bool(mask.any()):
                    group_losses.append(point_loss[row, mask].mean())
            rex_loss = torch.stack(group_losses).var(unbiased=False) if len(group_losses) > 1 else mean_loss.new_tensor(0.0)
            under_loss = mean_loss.new_tensor(0.0)
            if float(cfg.lambda_cold_lowsoc_underpred_loss) > 0.0:
                under_mask = hard_mask & valid_mask & pred[:, :, 0].lt(y[:, :, 0])
                if bool(under_mask.any()):
                    under_loss = F.smooth_l1_loss(
                        pred[:, :, 0][under_mask],
                        y[:, :, 0][under_mask],
                        beta=float(cfg.huber_beta),
                        reduction="mean",
                    )
            loss = mean_loss + float(cfg.lambda_rex) * rex_loss + float(cfg.lambda_cold_lowsoc_underpred_loss) * under_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            _split_state(model, state_next, active, states)
            losses.append(float(loss.detach().cpu()))
        loss_mean = float(np.mean(losses)) if losses else float("nan")
        history.append({"seed": int(seed), "epoch": int(ep), "train_loss": loss_mean})
        if ep == 1 or ep == int(cfg.epochs) or (int(cfg.print_every) > 0 and ep % int(cfg.print_every) == 0):
            print(f"[VALIDATION] {ep}/{cfg.epochs} [stateful_gru_gated_mlp] train_loss={loss_mean:.6f}", flush=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": int(seed),
            "config": {**asdict(cfg), "base_dir": str(cfg.base_dir), "raw_root": str(cfg.raw_root)},
            "feature_cols": feature_cols,
        },
        out_dir / f"{cfg.output_prefix}_seed{seed}_final.pt",
    )
    return model, pd.DataFrame(history)


@torch.no_grad()
def _predict_split(
    model: VcorrITGoalModel,
    frames: list[FrameTensor],
    split: str,
    seed: int,
    cfg: StatefulConfig,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for frame in frames:
        state = None
        preds = []
        n = int(frame.x.shape[0])
        for start in range(0, n, int(cfg.chunk_len)):
            x = frame.x[start : min(start + int(cfg.chunk_len), n)].unsqueeze(0)
            pred, _h, state = _stateful_forward(model, x, state)
            if state is not None:
                if model.recurrent == "lstm":
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            preds.append(pred[0, :, 0].detach().cpu().numpy())
        pred_np = np.concatenate(preds, axis=0)
        y_np = frame.y[:, 0].detach().cpu().numpy()
        start_eval = max(0, int(cfg.chunk_len) - 1)
        for idx in range(start_eval, n):
            rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "variant": "stateful_gru_gated_mlp_w50_tbptt",
                    "epoch": int(cfg.epochs),
                    "temperature": float(frame.temperature),
                    "temperature_C": float(frame.temperature),
                    "drive_cycle": frame.drive_cycle,
                    "file_name": frame.file_name,
                    "trajectory_id": frame.trajectory_id,
                    "end_index": int(frame.end_index[idx]),
                    "y_true": float(y_np[idx]),
                    "y_pred": float(pred_np[idx]),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = out["error"].abs()
    return out


def _summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (split, seed, variant, epoch, temp), g in pred.groupby(
        ["split", "seed", "variant", "epoch", "temperature_C"], dropna=False
    ):
        err = g["error"].to_numpy(np.float64)
        rows.append(
            {
                "split": split,
                "seed": int(seed),
                "variant": variant,
                "epoch": int(epoch),
                "temperature_C": float(temp),
                "n_windows": int(len(g)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(np.square(err))) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
            }
        )
    by_temp = pd.DataFrame(rows)
    test = by_temp[by_temp["split"].eq("test")].copy()
    piv = test.pivot_table(index=["seed", "variant", "epoch"], columns="temperature_C", values="MAE_pct", aggfunc="mean").reset_index()
    for col in [0.0, 25.0, 45.0]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv["max_target"] = piv[[0.0, 25.0, 45.0]].max(axis=1)
    piv["target_met"] = (piv[0.0] < 0.8) & (piv[25.0] < 0.7) & (piv[45.0] < 0.3)
    return by_temp, piv


def run(cfg: StatefulConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    configure_torch_runtime()
    out_dir = Path(cfg.base_dir).resolve() / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols, scaled = _prepare_frames(cfg, out_dir)
    train_frames = _as_frame_tensors(scaled["train"], feature_cols, int(cfg.chunk_len))
    test_frames = _as_frame_tensors(scaled["test"], feature_cols, int(cfg.chunk_len))
    if not train_frames or not test_frames:
        raise RuntimeError(f"Empty stateful frames: train={len(train_frames)} test={len(test_frames)}")
    all_history = []
    all_by_temp = []
    all_summary = []
    started = time.time()
    for seed in cfg.seeds:
        model, history = _train_one_seed(cfg, int(seed), feature_cols, train_frames, out_dir)
        all_history.append(history)
        pred_train = _predict_split(model, train_frames, "train", int(seed), cfg)
        pred_test = _predict_split(model, test_frames, "test", int(seed), cfg)
        pred = pd.concat([pred_train, pred_test], ignore_index=True)
        if bool(cfg.save_predictions):
            pred.to_csv(out_dir / f"{cfg.output_prefix}_seed{seed}_prediction_rows.csv.gz", index=False, compression="gzip")
        by_temp, summary = _summarize_predictions(pred)
        all_by_temp.append(by_temp)
        all_summary.append(summary)
        elapsed = time.time() - started
        print(f"[stateful] seed={seed} elapsed_sec={elapsed:.1f}", flush=True)
        print(summary.to_string(index=False), flush=True)
    history_df = pd.concat(all_history, ignore_index=True)
    by_temp_df = pd.concat(all_by_temp, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    history_df.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    by_temp_df.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    summary_df.to_csv(out_dir / f"{cfg.output_prefix}_test_summary.csv", index=False)
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "base_dir": str(cfg.base_dir),
                "raw_root": str(cfg.raw_root),
                "holdout": str(cfg.holdout).upper(),
                "train_profiles": train_profiles_for_holdout(cfg.holdout),
                "test_profiles": (str(cfg.holdout).upper(),),
                "feature_columns": feature_cols,
                "stateful_protocol": "causal GRU hidden state is carried across chronological 50-sample chunks within each trajectory and reset at trajectory boundaries",
                "validation_profiles": (),
                "uses_soc_input": False,
                "uses_profile_label_input": False,
                "uses_coulomb_counting": False,
                "checkpoint_selection": "last epoch only",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return by_temp_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Stateful GRU stream test for SOC80 w50 VALIDATION bottleneck.")
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    parser.add_argument("--holdout", default="VALIDATION")
    parser.add_argument("--feature-set", default="paper_g4_all_ema")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-print-every", type=int, default=10)
    args = parser.parse_args()
    holdout = str(args.holdout).upper()
    seeds = parse_seeds(args.seeds)
    prefix = args.output_prefix or (
        f"soc80_goal_w50_stateful_gru_stream_holdout{holdout.lower()}_"
        f"seeds{''.join(str(s) for s in seeds)}_e{int(args.epochs)}"
    )
    cfg = StatefulConfig(
        base_dir=Path(args.base_dir).resolve(),
        raw_root=Path(args.raw_root),
        holdout=holdout,
        feature_set=str(args.feature_set),
        output_prefix=prefix,
        seeds=seeds,
        epochs=int(args.epochs),
        chunk_len=int(args.chunk_len),
        hidden_size=int(args.hidden_size),
        print_every=int(args.batch_print_every),
    )
    run(cfg)


if __name__ == "__main__":
    main()
