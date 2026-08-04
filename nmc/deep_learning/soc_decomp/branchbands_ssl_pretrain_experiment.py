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
from torch.utils.data import DataLoader, Dataset

from .config import make_cfg
from .deep_no_leak_experiment import (
    DeepNoLeakTCN,
    add_derived_features,
    augmented_input_dim,
    augment_window_tensor,
    feature_columns,
    predict,
    temp_balanced_rex_loss,
)
from .extrapolation_robustness import temperature_balanced_loader
from .runtime import configure_torch_runtime, device
from .smoothq_retrain import (
    EXPERIMENTS,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .training import (
    attach_prediction_features,
    build_prediction_feature_lookup,
    make_scaled_frames_for_ablation,
)
from .variance_control import _overall_metrics, variance_by_temperature


SSL_TARGET_COLUMNS = [
    "V_hys_raw",
    "V_pol_raw",
    "V_residual_low",
    "V_residual_mid",
    "V_residual_high",
]

SSL_MASK_INPUT_COLUMNS = [
    "V_pol_raw",
    "V_hys_raw",
    "R0_x_V_pol",
    "T_x_V_pol",
    "V_pol_x_abs_dI",
    "V_pol_fast_raw",
    "V_pol_mid_raw",
    "V_pol_slow_raw",
    "V_residual_raw",
    "V_residual_low",
    "V_residual_mid",
    "V_residual_high",
    "V_residual_low_x_T",
    "V_residual_mid_x_absI",
    "V_residual_high_x_abs_dI",
]


@dataclass
class SSLBranchBandsConfig:
    base_dir: Path = Path(".")
    output_prefix: str = "ssl_branchbands_expA_w150_s3"
    experiment: str = "Exp A"
    feature_dir: str = "decomposed_features_train_temp_minus10_0_10_20_25_50"
    seed: int = 0
    window_len: int = 150
    stride: int = 3
    pretrain_epochs: int = 120
    finetune_epochs: int = 300
    batch_size: int = 1024
    lr: float = 8e-4
    ssl_lr: float = 8e-4
    weight_decay: float = 1e-4
    hidden_size: int = 128
    layers: int = 6
    kernel_size: int = 5
    norm_kind: str = "channel"
    dropout: float = 0.04
    lambda_rex: float = 2.0
    rex_group: str = "temperature_drive"
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    ssl_huber_beta: float = 0.05
    lambda_smooth: float = 0.0
    endpoint_loss_weight: float = 0.0
    lambda_worst: float = 0.0
    window_feature_mode: str = "delta_start_time"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 25
    save_predictions: bool = True


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_string(index=False) + "\n```"


def move_float(x: torch.Tensor) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def make_loader(ds: Dataset, cfg: SSLBranchBandsConfig, *, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def meta_for_frame(frame: dict, end: int) -> dict:
    soc_for_bin = float(frame["y_physical"][end])
    if soc_for_bin < 0.2:
        soc_bin = 0
    elif soc_for_bin <= 0.8:
        soc_bin = 1
    else:
        soc_bin = 2
    return {
        "file_name": frame["file_name"][end],
        "trajectory_id": frame["trajectory_id"][end],
        "end_index": int(frame["end_index"][end]),
        "temperature": float(frame["temperature"][end]),
        "drive_cycle": frame["drive_cycle"][end],
        "soc_bin": int(soc_bin),
    }


class SSLComponentWindowDataset(Dataset):
    def __init__(
        self,
        frames: list[pd.DataFrame],
        feature_cols: list[str],
        target_cols: list[str],
        mask_input_cols: list[str],
        window_len: int,
        stride: int,
        window_feature_mode: str,
    ):
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.mask_indices = [self.feature_cols.index(c) for c in mask_input_cols if c in self.feature_cols]
        self.target_indices = [self.feature_cols.index(c) for c in self.target_cols]
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.window_feature_mode = str(window_feature_mode)
        self.frames: list[dict] = []
        self.index: list[tuple[int, int, int]] = []
        for fi, frame in enumerate(frames):
            frame = frame.reset_index(drop=True)
            x = np.ascontiguousarray(frame[self.feature_cols].to_numpy(np.float32))
            frame_cache = {
                "x": x,
                "y_physical": np.ascontiguousarray(frame["SOC_physical"].to_numpy(np.float32)),
                "file_name": frame["file_name"].to_numpy(),
                "trajectory_id": frame["trajectory_id"].to_numpy(),
                "end_index": frame["end_index"].to_numpy(np.int64),
                "temperature": frame["temperature"].to_numpy(np.float32),
                "drive_cycle": frame["drive_cycle"].to_numpy(),
            }
            self.frames.append(frame_cache)
            n = len(frame)
            if n < self.window_len:
                continue
            for start in range(0, n - self.window_len + 1, self.stride):
                end = start + self.window_len - 1
                target = x[start : end + 1, :][:, self.target_indices]
                if np.isfinite(target).all() and np.isfinite(frame_cache["y_physical"][end]):
                    self.index.append((fi, start, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, start, end = self.index[idx]
        frame = self.frames[fi]
        x_full = frame["x"][start : end + 1]
        target = np.ascontiguousarray(x_full[:, self.target_indices])
        x_masked = np.array(x_full, copy=True)
        if self.mask_indices:
            x_masked[:, self.mask_indices] = 0.0
        x_tensor = augment_window_tensor(torch.from_numpy(x_masked), self.window_feature_mode, self.feature_cols)
        meta = meta_for_frame(frame, end)
        return x_tensor, torch.from_numpy(target), meta


class SequenceSOCWindowDataset(Dataset):
    def __init__(
        self,
        frames: list[pd.DataFrame],
        feature_cols: list[str],
        window_len: int,
        stride: int,
        window_feature_mode: str,
    ):
        self.feature_cols = list(feature_cols)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.window_feature_mode = str(window_feature_mode)
        self.frames: list[dict] = []
        self.index: list[tuple[int, int, int]] = []
        for fi, frame in enumerate(frames):
            frame = frame.reset_index(drop=True)
            frame_cache = {
                "x": np.ascontiguousarray(frame[self.feature_cols].to_numpy(np.float32)),
                "y_physical": np.ascontiguousarray(frame["SOC_physical"].to_numpy(np.float32)),
                "file_name": frame["file_name"].to_numpy(),
                "trajectory_id": frame["trajectory_id"].to_numpy(),
                "end_index": frame["end_index"].to_numpy(np.int64),
                "temperature": frame["temperature"].to_numpy(np.float32),
                "drive_cycle": frame["drive_cycle"].to_numpy(),
            }
            self.frames.append(frame_cache)
            n = len(frame)
            if n < self.window_len:
                continue
            for start in range(0, n - self.window_len + 1, self.stride):
                end = start + self.window_len - 1
                y = frame_cache["y_physical"][start : end + 1]
                if np.isfinite(y).all():
                    self.index.append((fi, start, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, start, end = self.index[idx]
        frame = self.frames[fi]
        x = frame["x"][start : end + 1]
        y = np.ascontiguousarray(frame["y_physical"][start : end + 1, None])
        x_tensor = augment_window_tensor(torch.from_numpy(x), self.window_feature_mode, self.feature_cols)
        meta = meta_for_frame(frame, end)
        return x_tensor, torch.from_numpy(y), meta


class EndpointSOCWindowDataset(SequenceSOCWindowDataset):
    def __getitem__(self, idx: int):
        x, y, meta = super().__getitem__(idx)
        return x, y[-1], meta


class SSLPretrainTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_size: int,
        layers: int,
        kernel_size: int,
        norm_kind: str,
        dropout: float,
    ):
        super().__init__()
        self.backbone = DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            layers=layers,
            kernel_size=kernel_size,
            norm_kind=norm_kind,
            dropout=dropout,
        )
        self.ssl_head = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_size, target_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.encode_sequence(x)
        return self.ssl_head(h).transpose(1, 2)


def copy_pretrained_encoder(pretrained: SSLPretrainTCN, finetune: DeepNoLeakTCN) -> None:
    finetune.input_proj.load_state_dict(pretrained.backbone.input_proj.state_dict())
    finetune.blocks.load_state_dict(pretrained.backbone.blocks.state_dict())


def run_ssl_branchbands(cfg: SSLBranchBandsConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir)
    configure_torch_runtime()
    set_seed(cfg.seed)

    base_cfg = make_cfg()
    base_cfg.base_dir = cfg.base_dir
    base_cfg.output_dir = cfg.base_dir
    base_cfg = experiment_cfg(base_cfg, cfg.experiment)
    if cfg.feature_dir:
        base_cfg.decomposed_dir = cfg.base_dir / cfg.feature_dir
    configure_strict_training(base_cfg)
    base_cfg.window_len = int(cfg.window_len)
    base_cfg.stride = int(cfg.stride)
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.lstm_lr = float(cfg.lr)
    base_cfg.lstm_weight_decay = float(cfg.weight_decay)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0

    lookup = load_smoothq_lookup(cfg.base_dir)
    frames = add_derived_features(load_relabelled_frames(base_cfg, cfg.experiment, lookup))
    feature_cols = feature_columns("branch_bands")
    missing_targets = [c for c in SSL_TARGET_COLUMNS if c not in feature_cols]
    if missing_targets:
        raise KeyError(f"SSL target columns must be selected feature columns: {missing_targets}")
    available = set().union(*(set(f.columns) for split in frames.values() for f in split))
    missing = [c for c in feature_cols + SSL_TARGET_COLUMNS if c not in available]
    if missing:
        raise KeyError(f"Missing feature/target columns: {missing}")
    scaled, _ = make_scaled_frames_for_ablation(frames, feature_cols)
    input_dim = augmented_input_dim(len(feature_cols), cfg.window_feature_mode)

    ssl_train_ds = SSLComponentWindowDataset(
        scaled["train"],
        feature_cols,
        SSL_TARGET_COLUMNS,
        SSL_MASK_INPUT_COLUMNS,
        cfg.window_len,
        cfg.stride,
        cfg.window_feature_mode,
    )
    soc_train_ds = SequenceSOCWindowDataset(
        scaled["train"],
        feature_cols,
        cfg.window_len,
        cfg.stride,
        cfg.window_feature_mode,
    )
    soc_test_ds = EndpointSOCWindowDataset(
        scaled["test"],
        feature_cols,
        cfg.window_len,
        1,
        cfg.window_feature_mode,
    )
    if len(ssl_train_ds) == 0 or len(soc_train_ds) == 0 or len(soc_test_ds) == 0:
        raise RuntimeError(f"Empty dataset ssl={len(ssl_train_ds)} train={len(soc_train_ds)} test={len(soc_test_ds)}")

    ssl_loader = make_loader(ssl_train_ds, cfg, shuffle=True)
    train_loader = temperature_balanced_loader(soc_train_ds, base_cfg, shuffle=True)
    test_loader = make_loader(soc_test_ds, cfg, shuffle=False)

    ssl_model = SSLPretrainTCN(
        input_dim=input_dim,
        target_dim=len(SSL_TARGET_COLUMNS),
        hidden_size=cfg.hidden_size,
        layers=cfg.layers,
        kernel_size=cfg.kernel_size,
        norm_kind=cfg.norm_kind,
        dropout=cfg.dropout,
    ).to(device)
    ssl_opt = torch.optim.AdamW(ssl_model.parameters(), lr=float(cfg.ssl_lr), weight_decay=float(cfg.weight_decay))
    ssl_history = []
    for ep in range(1, int(cfg.pretrain_epochs) + 1):
        ssl_model.train()
        losses = []
        for x, target, _meta in ssl_loader:
            pred = ssl_model(move_float(x))
            y = move_float(target)
            loss = F.smooth_l1_loss(pred, y, beta=float(cfg.ssl_huber_beta))
            ssl_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ssl_model.parameters(), 1.0)
            ssl_opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": ep, "ssl_loss": float(np.mean(losses))}
        ssl_history.append(row)
        if ep == 1 or ep == int(cfg.pretrain_epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(f"SSL epoch={ep} loss={row['ssl_loss']:.6f}", flush=True)

    soc_model = DeepNoLeakTCN(
        input_dim=input_dim,
        hidden_size=cfg.hidden_size,
        layers=cfg.layers,
        kernel_size=cfg.kernel_size,
        norm_kind=cfg.norm_kind,
        dropout=cfg.dropout,
    ).to(device)
    copy_pretrained_encoder(ssl_model, soc_model)
    soc_opt = torch.optim.AdamW(soc_model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    finetune_history = []
    for ep in range(1, int(cfg.finetune_epochs) + 1):
        soc_model.train()
        losses = []
        mean_losses = []
        rex_losses = []
        smooth_losses = []
        by_group_all: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            pred = soc_model.forward_sequence(move_float(x))
            yy = move_float(y)
            loss, mean_loss, rex_loss, smooth_loss, by_group = temp_balanced_rex_loss(
                pred,
                yy,
                meta,
                cfg.lambda_rex,
                cfg.rex_group,
                cfg.loss_kind,
                cfg.huber_beta,
                cfg.lambda_smooth,
                cfg.endpoint_loss_weight,
                cfg.lambda_worst,
            )
            soc_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(soc_model.parameters(), 1.0)
            soc_opt.step()
            losses.append(float(loss.detach().cpu()))
            mean_losses.append(float(mean_loss.detach().cpu()))
            rex_losses.append(float(rex_loss.detach().cpu()))
            smooth_losses.append(float(smooth_loss.detach().cpu()))
            for k, v in by_group.items():
                by_group_all.setdefault(k, []).append(v)
        row = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_abs_loss": float(np.mean(mean_losses)),
            "rex_var": float(np.mean(rex_losses)),
            "smooth_delta_loss": float(np.mean(smooth_losses)),
        }
        for k, vals in by_group_all.items():
            safe = str(k).replace(".", "p").replace("-", "N")
            row[f"train_loss_group_{safe}"] = float(np.mean(vals))
        finetune_history.append(row)
        if ep == 1 or ep == int(cfg.finetune_epochs) or ep % max(1, int(cfg.print_every)) == 0:
            print(
                f"SOC epoch={ep} loss={row['loss']:.6f} mean={row['mean_abs_loss']:.6f} "
                f"rex={row['rex_var']:.8f}",
                flush=True,
            )

    model_name = f"SSLBranchBands_tcn_w{cfg.window_len}_s{cfg.stride}_seed{cfg.seed}"
    pred = predict(soc_model, test_loader)
    lookup_features = build_prediction_feature_lookup(frames)
    pred = attach_prediction_features(pred.assign(split="test", ablation=model_name), lookup_features, ablation_name=model_name, target_label="physical")
    pred["experiment"] = cfg.experiment
    pred["seed"] = int(cfg.seed)
    pred["ssl_pretrained"] = True
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

    prefix = str(cfg.output_prefix)
    if cfg.save_predictions:
        pred.to_csv(cfg.base_dir / f"{prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    pd.DataFrame(ssl_history).to_csv(cfg.base_dir / f"{prefix}_ssl_history.csv", index=False)
    pd.DataFrame(finetune_history).to_csv(cfg.base_dir / f"{prefix}_finetune_history.csv", index=False)
    overall.to_csv(cfg.base_dir / f"{prefix}_overall.csv", index=False)
    by_temp.to_csv(cfg.base_dir / f"{prefix}_by_temperature.csv", index=False)
    focus.to_csv(cfg.base_dir / f"{prefix}_focus.csv", index=False)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "model_name": model_name,
        "feature_set": "branch_bands",
        "feature_columns": feature_cols,
        "input_feature_dim": int(input_dim),
        "ssl_target_columns": SSL_TARGET_COLUMNS,
        "ssl_mask_input_columns": [c for c in SSL_MASK_INPUT_COLUMNS if c in feature_cols],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_current_integration": False,
        "current_policy": "current is used only as instantaneous excitation; it is not integrated into SOC state",
        "pretrain_uses_soc_labels": False,
        "pretrain_future_current_input": False,
        "train_windows": int(len(soc_train_ds)),
        "ssl_train_windows": int(len(ssl_train_ds)),
        "test_windows": int(len(soc_test_ds)),
    }
    write_json(cfg.base_dir / f"{prefix}_metadata.json", metadata)
    report_lines = [
        "# SSL BranchBands Result",
        "",
        f"- Model: `{model_name}`",
        f"- Experiment: `{cfg.experiment}`",
        f"- Train temps: `{EXPERIMENTS[cfg.experiment]['train_temps']}`",
        "- Eval temps: `N10, 0, 10, 20, 25, 30, 40, 50`",
        "- Strict: no SOC input, no cumulative Ah input, no explicit current-integration SOC update.",
        "- SSL pretraining masks target component columns in the input and reconstructs voltage decomposition components.",
        "",
        "## Overall",
        "",
        table_text(overall),
        "",
        "## Focus",
        "",
        table_text(focus),
        "",
        "## By Temperature",
        "",
        table_text(by_temp[["temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio"]]),
    ]
    (cfg.base_dir / f"{prefix}_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("Overall:")
    print(overall.to_string(index=False))
    print("Focus:")
    print(focus.to_string(index=False))
    print("By temperature:")
    print(by_temp[["temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio"]].to_string(index=False))
    return {
        "prediction_rows": pred,
        "overall": overall,
        "by_temperature": by_temp,
        "focus": focus,
        "ssl_history": pd.DataFrame(ssl_history),
        "finetune_history": pd.DataFrame(finetune_history),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BranchBands TCN with SSL voltage-component pretraining under strict NoCC constraints.")
    p.add_argument("--base-dir", type=Path, default=Path("."))
    p.add_argument("--output-prefix", default="ssl_branchbands_expA_w150_s3")
    p.add_argument("--experiment", default="Exp A")
    p.add_argument("--feature-dir", default="decomposed_features_train_temp_minus10_0_10_20_25_50")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-len", type=int, default=150)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--pretrain-epochs", type=int, default=120)
    p.add_argument("--finetune-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--ssl-lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--norm-kind", choices=["channel", "group"], default="channel")
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--rex-group", default="temperature_drive")
    p.add_argument("--loss-kind", choices=["mae", "huber", "mse"], default="huber")
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--ssl-huber-beta", type=float, default=0.05)
    p.add_argument("--window-feature-mode", default="delta_start_time")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--no-save-predictions", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SSLBranchBandsConfig(
        base_dir=args.base_dir,
        output_prefix=args.output_prefix,
        experiment=args.experiment,
        feature_dir=args.feature_dir,
        seed=args.seed,
        window_len=args.window_len,
        stride=args.stride,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ssl_lr=args.ssl_lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        layers=args.layers,
        kernel_size=args.kernel_size,
        norm_kind=args.norm_kind,
        dropout=args.dropout,
        lambda_rex=args.lambda_rex,
        rex_group=args.rex_group,
        loss_kind=args.loss_kind,
        huber_beta=args.huber_beta,
        ssl_huber_beta=args.ssl_huber_beta,
        window_feature_mode=args.window_feature_mode,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        print_every=args.print_every,
        save_predictions=not bool(args.no_save_predictions),
    )
    run_ssl_branchbands(cfg)


if __name__ == "__main__":
    main()
