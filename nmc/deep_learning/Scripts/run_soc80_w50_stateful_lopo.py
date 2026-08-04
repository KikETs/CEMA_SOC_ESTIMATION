#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_decomp.nmc_vcorr_it_train_dst_selector_run import (
    TrainDSTSelectorConfig,
    _cached_feature_frames,
    _cached_find_csv_files,
    _cached_r0_by_temperature,
    _cached_scaled_frames_for_ablation,
    _filter_scaled_frames_by_temperatures,
    _selected_feature_columns,
)
from soc_decomp.runtime import configure_torch_runtime, device


PROFILE_SET = ("VALIDATION", "DST", "FUDS", "US06")


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model_kind: str = "gru"
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.04
    lr: float = 8e-4
    weight_decay: float = 2e-4
    weight_0: float = 1.0
    weight_25: float = 1.0
    weight_45: float = 1.0
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    lambda_delta: float = 0.0


CANDIDATES = (
    Candidate("stateful_gru_all_ema_h128_l2", "paper_g4_all_ema"),
    Candidate("stateful_gru_all_ema_start_h128_l2", "paper_g4_all_ema_start"),
    Candidate("stateful_gru_eqdyn_h128_l2", "paper_g4_eqdyn"),
    Candidate("stateful_gru_eqdyn_start_h128_l2", "paper_g4_eqdyn_start"),
    Candidate("stateful_gru_eqdyn_h64_l2_w0x2", "paper_g4_eqdyn", hidden_size=64, weight_0=2.0, weight_25=1.5),
    Candidate("stateful_gru_all_ema_h64_l2_w0x2", "paper_g4_all_ema", hidden_size=64, weight_0=2.0, weight_25=1.5),
)


class StatefulSOCModel(nn.Module):
    def __init__(self, input_dim: int, candidate: Candidate):
        super().__init__()
        hidden = int(candidate.hidden_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        rnn_cls = nn.GRU if candidate.model_kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            hidden,
            hidden,
            num_layers=int(candidate.layers),
            batch_first=True,
            dropout=float(candidate.dropout) if int(candidate.layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(float(candidate.dropout)),
            nn.Linear(hidden, 1),
        )

    def forward_sequence(self, x: torch.Tensor, state=None):
        z = self.input_proj(x)
        out, state = self.rnn(z, state)
        pred = torch.sigmoid(self.head(self.norm(out)))
        return pred, state


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def train_profiles_for_holdout(holdout: str) -> tuple[str, ...]:
    holdout = holdout.upper()
    return tuple(profile for profile in PROFILE_SET if profile != holdout)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def candidate_map() -> dict[str, Candidate]:
    return {candidate.name: candidate for candidate in CANDIDATES}


def detach_state(state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    return tuple(s.detach() for s in state)


def frame_to_tensors(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(frame[feature_cols].to_numpy(np.float32), device=device)[None, :, :]
    y = torch.as_tensor(frame["SOC_physical"].to_numpy(np.float32), device=device)[None, :, None]
    return x, y


def point_loss(pred: torch.Tensor, y: torch.Tensor, candidate: Candidate) -> torch.Tensor:
    if candidate.loss_kind == "mae":
        return torch.abs(pred - y)
    if candidate.loss_kind == "mse":
        return (pred - y) ** 2
    return F.smooth_l1_loss(pred, y, beta=float(candidate.huber_beta), reduction="none")


def temperature_weight(frame: pd.DataFrame, candidate: Candidate) -> float:
    temp = float(frame["temperature"].iloc[0])
    if np.isclose(temp, 0.0, atol=0.75):
        return float(candidate.weight_0)
    if np.isclose(temp, 25.0, atol=0.75):
        return float(candidate.weight_25)
    if np.isclose(temp, 45.0, atol=0.75):
        return float(candidate.weight_45)
    return 1.0


def train_one_epoch(
    model: StatefulSOCModel,
    optimizer: torch.optim.Optimizer,
    train_frames: list[pd.DataFrame],
    feature_cols: list[str],
    candidate: Candidate,
    bptt_len: int,
) -> dict[str, float]:
    random.shuffle(train_frames)
    model.train()
    losses: list[float] = []
    maes: list[float] = []
    for frame in train_frames:
        x_all, y_all = frame_to_tensors(frame, feature_cols)
        state = None
        frame_weight = temperature_weight(frame, candidate)
        for start in range(0, x_all.shape[1], int(bptt_len)):
            x = x_all[:, start : start + int(bptt_len), :]
            y = y_all[:, start : start + int(bptt_len), :]
            if x.shape[1] == 0:
                continue
            pred, state = model.forward_sequence(x, state)
            loss = point_loss(pred, y, candidate).mean() * frame_weight
            if float(candidate.lambda_delta) > 0.0 and pred.shape[1] > 1:
                delta_loss = torch.abs((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])).mean()
                loss = loss + float(candidate.lambda_delta) * delta_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = detach_state(state)
            losses.append(float(loss.detach().cpu()))
            maes.append(float(torch.abs(pred.detach() - y).mean().cpu()))
    return {"loss": float(np.mean(losses)), "mae": float(np.mean(maes))}


@torch.no_grad()
def predict_frames(
    model: StatefulSOCModel,
    frames: list[pd.DataFrame],
    feature_cols: list[str],
    candidate_name: str,
    seed: int,
    split: str,
    bptt_len: int,
    warmup: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for frame in frames:
        f = frame.reset_index(drop=True)
        x_all = torch.as_tensor(f[feature_cols].to_numpy(np.float32), device=device)
        preds = []
        state = None
        for start in range(0, len(f), int(bptt_len)):
            x = x_all[start : start + int(bptt_len)][None, :, :]
            pred, state = model.forward_sequence(x, state)
            preds.append(pred.detach().cpu().numpy()[0, :, 0])
        y_pred = np.concatenate(preds, axis=0)
        out = pd.DataFrame(
            {
                "split": split,
                "candidate": candidate_name,
                "seed": int(seed),
                "trajectory_id": f["trajectory_id"].to_numpy(),
                "file_name": f["file_name"].to_numpy() if "file_name" in f.columns else f["trajectory_id"].to_numpy(),
                "drive_cycle": f["drive_cycle"].to_numpy(),
                "temperature": f["temperature"].to_numpy(),
                "end_index": f["end_index"].to_numpy(),
                "y_true": f["SOC_physical"].to_numpy(np.float32),
                "y_pred": y_pred.astype(np.float32),
            }
        )
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        out = out[out["end_index"].astype(int) >= int(warmup)].copy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metrics_by_temperature(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for (split, candidate, seed, temp), group in pred.groupby(["split", "candidate", "seed", "temperature"], dropna=False):
        rows.append(
            {
                "split": split,
                "candidate": candidate,
                "seed": int(seed),
                "temperature_C": float(temp),
                "n": int(len(group)),
                "MAE_pct": float(group["abs_error"].mean() * 100.0),
                "Bias_pct": float(group["error"].mean() * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(np.square(group["error"].to_numpy()))) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def make_data_cfg(args: argparse.Namespace, holdout: str, feature_set: str) -> TrainDSTSelectorConfig:
    return TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.tag),
        train_profiles=train_profiles_for_holdout(holdout),
        valid_profiles=("NONE",),
        test_profiles=(holdout.upper(),),
        train_temperatures=(),
        valid_temperatures=(),
        test_temperatures=(),
        window_len=int(args.bptt_len),
        stride=3,
        feature_set=str(feature_set),
        valid_split_mode="profile",
    )


def load_scaled_frames(args: argparse.Namespace, holdout: str, feature_set: str):
    cfg = make_data_cfg(args, holdout, feature_set)
    files = _cached_find_csv_files(Path(cfg.raw_root))
    r0_df = _cached_r0_by_temperature(cfg, files)
    frames = _cached_feature_frames(cfg, files, r0_df)
    feature_cols = _selected_feature_columns(feature_set)
    scaled, scaler = _cached_scaled_frames_for_ablation(frames, feature_cols)
    scaled = _filter_scaled_frames_by_temperatures(
        scaled,
        train_temperatures=(),
        valid_temperatures=(),
        test_temperatures=(),
        name=f"{holdout}:{feature_set}",
    )
    return cfg, feature_cols, scaled, scaler


def run_job(args: argparse.Namespace, holdout: str, candidate: Candidate, seed: int) -> dict[str, pd.DataFrame]:
    set_seed(seed)
    cfg, feature_cols, scaled, _scaler = load_scaled_frames(args, holdout, candidate.feature_set)
    model = StatefulSOCModel(len(feature_cols), candidate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(candidate.lr), weight_decay=float(candidate.weight_decay))
    history = []
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        row = train_one_epoch(model, optimizer, list(scaled["train"]), feature_cols, candidate, int(args.bptt_len))
        row.update({"candidate": candidate.name, "holdout": holdout.upper(), "seed": int(seed), "epoch": int(epoch)})
        history.append(row)
        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.print_every)) == 0:
            print(
                f"[{holdout.upper()}] [{epoch}/{int(args.epochs)}] "
                f"{candidate.name} loss={row['loss']:.5f} train_mae={row['mae']*100.0:.3f}% "
                f"elapsed={format_eta(time.time() - started)}",
                flush=True,
            )
    warmup = max(0, int(args.bptt_len) - 1)
    pred_test = predict_frames(
        model,
        scaled["test"],
        feature_cols,
        candidate.name,
        seed,
        "test",
        int(args.bptt_len),
        warmup,
    )
    metrics = metrics_by_temperature(pred_test)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {"history": pd.DataFrame(history), "pred_test": pred_test, "metrics": metrics, "state_dict": state_dict}


def main() -> None:
    parser = argparse.ArgumentParser(description="LOPO online stateful SOC model with 50-step truncated BPTT.")
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--raw-root", default="nmc_soc80_train_nofloor_qmax")
    parser.add_argument("--holdouts", default="VALIDATION")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--candidates", default="all")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--bptt-len", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--tag", default="stateful_w50_screen_v1")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configure_torch_runtime()
    out_dir = Path(args.base_dir) / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = candidate_map()
    selected_names = tuple(cmap) if str(args.candidates).lower() == "all" else parse_csv(args.candidates)
    missing = [name for name in selected_names if name not in cmap]
    if missing:
        raise SystemExit(f"unknown candidates: {missing}")

    all_metrics = []
    all_history = []
    manifest = []
    jobs = [(holdout.upper(), cmap[name], seed) for holdout in parse_csv(args.holdouts) for name in selected_names for seed in parse_seeds(args.seeds)]
    started = time.time()
    for idx, (holdout, candidate, seed) in enumerate(jobs, start=1):
        prefix = f"soc80_goal_w50_{args.tag}_{candidate.name}_holdout{holdout.lower()}_seed{seed}_e{int(args.epochs)}"
        summary_path = out_dir / f"{prefix}_test_summary.csv"
        if args.skip_existing and summary_path.exists():
            print(f"[skip] {idx}/{len(jobs)} {candidate.name} {holdout} seed={seed}", flush=True)
            metrics = pd.read_csv(summary_path)
            all_metrics.append(metrics.assign(candidate=candidate.name, holdout=holdout, seed=int(seed)))
            continue
        elapsed = time.time() - started
        per_job = elapsed / max(1, idx - 1) if idx > 1 else 0.0
        print(
            f"[run] {idx}/{len(jobs)} {candidate.name} {holdout} seed={seed} "
            f"elapsed={format_eta(elapsed)} eta={format_eta(per_job * (len(jobs) - idx + 1))}",
            flush=True,
        )
        result = run_job(args, holdout, candidate, int(seed))
        metrics = result["metrics"].copy()
        metrics["holdout"] = holdout
        metrics["feature_set"] = candidate.feature_set
        metrics["model_kind"] = candidate.model_kind
        metrics["hidden_size"] = int(candidate.hidden_size)
        metrics["layers"] = int(candidate.layers)
        metrics["bptt_len"] = int(args.bptt_len)
        metrics.to_csv(summary_path, index=False)
        result["history"].to_csv(out_dir / f"{prefix}_history.csv", index=False)
        result["pred_test"].to_csv(out_dir / f"{prefix}_test_prediction_rows.csv.gz", index=False, compression="gzip")
        torch.save(
            {
                "candidate": candidate.name,
                "holdout": holdout,
                "seed": int(seed),
                "epochs": int(args.epochs),
                "bptt_len": int(args.bptt_len),
                "state_dict": result["state_dict"],
            },
            out_dir / f"{prefix}_final_weights.pt",
        )
        all_metrics.append(metrics)
        all_history.append(result["history"])
        manifest.append(
            {
                "prefix": prefix,
                "holdout": holdout,
                "seed": int(seed),
                **asdict(candidate),
                "bptt_len": int(args.bptt_len),
                "summary_path": summary_path.name,
                "test_predictions": f"{prefix}_test_prediction_rows.csv.gz",
            }
        )

    metrics_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    seed_path = out_dir / f"soc80_goal_w50_{args.tag}_seed_summary.csv"
    if len(metrics_all):
        piv = metrics_all[metrics_all["split"].eq("test")].pivot_table(
            index=["candidate", "holdout", "seed", "feature_set", "model_kind", "hidden_size", "layers", "bptt_len"],
            columns="temperature_C",
            values="MAE_pct",
            aggfunc="mean",
        ).reset_index()
        for temp in (0.0, 25.0, 45.0):
            if temp not in piv.columns:
                piv[temp] = np.nan
        piv = piv.rename(columns={0.0: "mae_0C", 25.0: "mae_25C", 45.0: "mae_45C"})
        piv["mae_mean_temp"] = piv[["mae_0C", "mae_25C", "mae_45C"]].mean(axis=1)
        piv.sort_values(["holdout", "mae_0C", "candidate"]).to_csv(seed_path, index=False)
        agg = (
            piv.groupby(["candidate", "holdout", "feature_set", "model_kind", "hidden_size", "layers", "bptt_len"], as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                mae_0C_mean=("mae_0C", "mean"),
                mae_0C_std=("mae_0C", "std"),
                mae_0C_min=("mae_0C", "min"),
                mae_0C_max=("mae_0C", "max"),
                mae_25C_mean=("mae_25C", "mean"),
                mae_45C_mean=("mae_45C", "mean"),
            )
            .sort_values(["holdout", "mae_0C_mean", "mae_0C_max"])
        )
    else:
        piv = pd.DataFrame()
        piv.to_csv(seed_path, index=False)
        agg = pd.DataFrame()
    agg_path = out_dir / f"soc80_goal_w50_{args.tag}_agg_summary.csv"
    agg.to_csv(agg_path, index=False)
    pd.DataFrame(manifest).to_csv(out_dir / f"soc80_goal_w50_{args.tag}_manifest.csv", index=False)
    metadata = {
        "tag": str(args.tag),
        "raw_root": str(Path(args.raw_root)),
        "bptt_len": int(args.bptt_len),
        "online_stateful": True,
        "state_reset": "per trajectory",
        "state_carried_between_chunks": True,
        "uses_soc_input": False,
        "uses_cumulative_current": False,
        "warmup_rows_excluded_from_metrics": max(0, int(args.bptt_len) - 1),
        "candidate_names": list(selected_names),
    }
    (out_dir / f"soc80_goal_w50_{args.tag}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] seed_summary={seed_path}", flush=True)
    print(f"[done] agg_summary={agg_path}", flush=True)
    if len(agg):
        print(agg.head(40).to_string(index=False), flush=True)
if __name__ == "__main__":
    main()
