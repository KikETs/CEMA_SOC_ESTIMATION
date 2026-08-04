from __future__ import annotations

from dataclasses import dataclass
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
from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .training import make_scaled_frames_for_ablation
from .variance_control import R5_GATED_FEATURES, _overall_metrics, variance_by_temperature
from .deep_no_leak_experiment import add_derived_features, feature_columns


@dataclass
class OnlineStatefulConfig:
    base_dir: Path = Path(".")
    output_prefix: str = "online_stateful"
    experiment: str = "Exp D"
    seed: int = 0
    model_kind: str = "gru"
    feature_set: str = "extended"
    epochs: int = 500
    lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.04
    loss_kind: str = "mae"
    huber_beta: float = 0.02
    lambda_smooth: float = 0.0
    lambda_mono: float = 0.0
    eval_chunk_len: int = 256
    print_every: int = 50


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class OnlineGRUSOC(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.04):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=int(layers),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )

    def forward_sequence(self, x, state=None):
        z = self.input_proj(x)
        out, state = self.rnn(z, state)
        pred = torch.sigmoid(self.head(self.norm(out)))
        return pred, state


class OnlineLSTMSOC(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.04):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=int(layers),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )

    def forward_sequence(self, x, state=None):
        z = self.input_proj(x)
        out, state = self.rnn(z, state)
        pred = torch.sigmoid(self.head(self.norm(out)))
        return pred, state


def make_model(cfg: OnlineStatefulConfig, input_dim: int):
    if cfg.model_kind == "gru":
        return OnlineGRUSOC(input_dim, cfg.hidden_size, cfg.layers, cfg.dropout)
    if cfg.model_kind == "lstm":
        return OnlineLSTMSOC(input_dim, cfg.hidden_size, cfg.layers, cfg.dropout)
    raise ValueError(f"Unknown model_kind={cfg.model_kind}")


def pointwise_loss(pred: torch.Tensor, y: torch.Tensor, loss_kind: str, huber_beta: float):
    if loss_kind == "mae":
        return torch.abs(pred - y)
    if loss_kind == "huber":
        return F.smooth_l1_loss(pred, y, beta=float(huber_beta), reduction="none")
    if loss_kind == "mse":
        return (pred - y) ** 2
    raise ValueError(f"Unknown loss_kind={loss_kind}")


def frame_tensors(frame: pd.DataFrame, cols: list[str]):
    x = torch.as_tensor(frame[cols].to_numpy(np.float32)[None, :, :], device=device)
    y = torch.as_tensor(frame["SOC_physical"].to_numpy(np.float32)[None, :, None], device=device)
    return x, y


def sequence_loss(pred: torch.Tensor, y: torch.Tensor, cfg: OnlineStatefulConfig):
    loss = torch.mean(pointwise_loss(pred, y, cfg.loss_kind, cfg.huber_beta))
    if float(cfg.lambda_smooth) > 0.0 and pred.size(1) > 1:
        smooth = torch.mean(torch.abs((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])))
        loss = loss + float(cfg.lambda_smooth) * smooth
    else:
        smooth = loss.new_tensor(0.0)
    if float(cfg.lambda_mono) > 0.0 and pred.size(1) > 1:
        mono = F.relu(pred[:, 1:] - pred[:, :-1]).mean()
        loss = loss + float(cfg.lambda_mono) * mono
    else:
        mono = loss.new_tensor(0.0)
    mae = torch.mean(torch.abs(pred - y))
    return loss, mae, smooth, mono


@torch.no_grad()
def predict_streaming(model, frames: list[pd.DataFrame], cols: list[str], model_name: str, chunk_len: int):
    model.eval()
    rows = []
    chunk_len = max(1, int(chunk_len))
    for frame in frames:
        f = frame.reset_index(drop=True)
        x_all = torch.as_tensor(f[cols].to_numpy(np.float32), device=device)
        preds = []
        state = None
        for start in range(0, len(f), chunk_len):
            x = x_all[start:start + chunk_len][None, :, :]
            pred, state = model.forward_sequence(x, state)
            preds.append(pred.detach().cpu().numpy()[0, :, 0])
        y_pred = np.concatenate(preds, axis=0)
        df = pd.DataFrame({
            "model_name": model_name,
            "target_label": "physical",
            "trajectory_id": f["trajectory_id"].to_numpy(),
            "file_name": f["file_name"].to_numpy() if "file_name" in f.columns else f["trajectory_id"].to_numpy(),
            "drive_cycle": f["drive_cycle"].to_numpy(),
            "temperature": f["temperature"].to_numpy(),
            "temperature_C": f["temperature"].to_numpy(),
            "end_index": f["end_index"].to_numpy(),
            "y_true": f["SOC_physical"].to_numpy(np.float32),
            "y_pred": y_pred,
        })
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = np.abs(df["error"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_online_stateful(cfg: OnlineStatefulConfig):
    cfg.base_dir = Path(cfg.base_dir)
    configure_torch_runtime()
    set_seed(cfg.seed)

    base_cfg = make_cfg()
    base_cfg.output_dir = cfg.base_dir
    base_cfg.base_dir = cfg.base_dir
    base_cfg = experiment_cfg(base_cfg, cfg.experiment)
    configure_strict_training(base_cfg)

    lookup = load_smoothq_lookup(cfg.base_dir)
    frames = add_derived_features(load_relabelled_frames(base_cfg, cfg.experiment, lookup))
    cols = feature_columns(cfg.feature_set)
    scaled, _ = make_scaled_frames_for_ablation(frames, cols)

    model_name = f"OnlineStateful_{cfg.model_kind}_{cfg.feature_set}_h{cfg.hidden_size}_l{cfg.layers}_seed{cfg.seed}"
    model = make_model(cfg, len(cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    train_frames = list(scaled["train"])
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        random.shuffle(train_frames)
        model.train()
        losses = []
        maes = []
        smoothes = []
        monos = []
        by_temp = {}
        for frame in train_frames:
            x, y = frame_tensors(frame, cols)
            pred, _ = model.forward_sequence(x, None)
            loss, mae, smooth, mono = sequence_loss(pred, y, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            maes.append(float(mae.detach().cpu()))
            smoothes.append(float(smooth.detach().cpu()))
            monos.append(float(mono.detach().cpu()))
            temp = float(frame["temperature"].iloc[0])
            by_temp.setdefault(temp, []).append(float(mae.detach().cpu()))
        row = {
            "model_name": model_name,
            "experiment": cfg.experiment,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mae_loss": float(np.mean(maes)),
            "smooth_loss": float(np.mean(smoothes)),
            "mono_loss": float(np.mean(monos)),
        }
        for temp, vals in by_temp.items():
            row[f"train_mae_temp_{temp:g}"] = float(np.mean(vals))
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(
                f"{model_name} epoch={ep} loss={row['loss']:.5f} "
                f"mae={row['mae_loss']:.5f} smooth={row['smooth_loss']:.5f} mono={row['mono_loss']:.5f}",
                flush=True,
            )

    pred = predict_streaming(model, scaled["test"], cols, model_name, cfg.eval_chunk_len)
    pred["experiment"] = cfg.experiment
    pred["seed"] = int(cfg.seed)
    overall = _overall_metrics(pred)
    overall["experiment"] = cfg.experiment
    by_temp = variance_by_temperature(pred)
    by_temp["experiment"] = cfg.experiment
    omitted_temp = float(EXPERIMENTS[cfg.experiment]["omitted_temp_C"])
    omitted = by_temp[np.isclose(by_temp["temperature_C"].astype(float), omitted_temp)]
    seen = by_temp[~np.isclose(by_temp["temperature_C"].astype(float), omitted_temp)]
    focus = pd.DataFrame([{
        "experiment": cfg.experiment,
        "model_name": model_name,
        "omitted_temperature_C": omitted_temp,
        "omitted_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
        "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
        "overall_MAE_pct": float(overall["MAE_pct"].iloc[0]) if len(overall) else np.nan,
        "worst_temperature_MAE_pct": float(by_temp["MAE_pct"].max()) if len(by_temp) else np.nan,
    }])

    prefix = cfg.output_prefix
    pred.to_csv(cfg.base_dir / f"{prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    pd.DataFrame(history).to_csv(cfg.base_dir / f"{prefix}_history.csv", index=False)
    overall.to_csv(cfg.base_dir / f"{prefix}_overall.csv", index=False)
    by_temp.to_csv(cfg.base_dir / f"{prefix}_by_temperature.csv", index=False)
    focus.to_csv(cfg.base_dir / f"{prefix}_focus.csv", index=False)
    metadata = {
        **cfg.__dict__,
        "base_dir": str(cfg.base_dir),
        "model_name": model_name,
        "feature_columns": cols,
        "forbidden_input_tokens": ["SOC", "soc", "cumulative", "Q_ref", "q_cutoff"],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_current_integration": False,
        "online_causal_streaming": True,
        "state_carried_between_steps": "hidden_state_only",
        "state_reset": "per_trajectory",
        "train_trajectories": int(len(scaled["train"])),
        "test_trajectories": int(len(scaled["test"])),
    }
    (cfg.base_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Overall:")
    print(overall.to_string(index=False))
    print("Focus:")
    print(focus.to_string(index=False))
    return {"pred": pred, "overall": overall, "by_temperature": by_temp, "focus": focus, "history": pd.DataFrame(history)}


def parse_args():
    p = argparse.ArgumentParser(description="Online causal stateful SOC model without SOC/cumulative inputs or current integration.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--output-prefix", default="online_stateful")
    p.add_argument("--experiment", default="Exp D")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-kind", choices=["gru", "lstm"], default="gru")
    p.add_argument("--feature-set", choices=["base", "extended", "bands", "voltage_only", "filtered"], default="extended")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--loss-kind", choices=["mae", "huber", "mse"], default="mae")
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--lambda-mono", type=float, default=0.0)
    p.add_argument("--eval-chunk-len", type=int, default=256)
    p.add_argument("--print-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OnlineStatefulConfig(
        base_dir=Path(args.base_dir),
        output_prefix=args.output_prefix,
        experiment=args.experiment,
        seed=args.seed,
        model_kind=args.model_kind,
        feature_set=args.feature_set,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
        loss_kind=args.loss_kind,
        huber_beta=args.huber_beta,
        lambda_smooth=args.lambda_smooth,
        lambda_mono=args.lambda_mono,
        eval_chunk_len=args.eval_chunk_len,
        print_every=args.print_every,
    )
    run_online_stateful(cfg)


if __name__ == "__main__":
    main()
