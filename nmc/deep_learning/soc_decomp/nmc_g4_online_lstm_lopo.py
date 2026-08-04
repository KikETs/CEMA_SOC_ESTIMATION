from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_train_dst_selector_run import TrainDSTSelectorConfig, _selected_feature_columns
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


@dataclass
class OnlineG4LOPOConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = "lopo_g4_online_lstm"
    seed: int = 0
    train_profiles: tuple[str, ...] = ("VALIDATION", "DST", "FUDS")
    valid_profiles: tuple[str, ...] = ()
    test_profiles: tuple[str, ...] = ("US06",)
    feature_set: str = "paper_g4_all_ema"
    model_kind: str = "plain"
    epochs: int = 200
    hidden_size: int = 128
    layers: int = 1
    dropout: float = 0.04
    lr: float = 8e-4
    weight_decay: float = 1e-4
    chunk_len: int = 512
    eval_chunk_len: int = 1024
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    weight_0: float = 0.8
    weight_25: float = 2.2
    weight_45: float = 1.0
    lambda_smooth: float = 0.0
    lambda_mono: float = 0.0
    checkpoint_selector: str = "final"
    soc_weight_mode: str = "none"
    soc_weight_strength: float = 1.0
    soc_weight_bins: int = 10
    soc_weight_clip: float = 4.0
    soc_edge_low: float = 0.12
    soc_edge_high: float = 0.72
    residual_limit: float = 0.25
    print_every: int = 20
    save_predictions: bool = False
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


class OnlineLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, layers: int, dropout: float):
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

    def forward_sequence(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        z = self.input_proj(x)
        h, state = self.rnn(z, state)
        pred = torch.sigmoid(self.head(self.norm(h)))
        return pred, state


class OnlineTempHeadLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        dropout: float,
        mode: str,
    ):
        super().__init__()
        if "T" not in feature_cols:
            raise RuntimeError(f"Temperature-head online LSTM requires T, got {feature_cols}")
        self.temp_idx = int(feature_cols.index("T"))
        self.mode = str(mode)
        if self.mode not in {"hard", "moe"}:
            raise ValueError(f"Unknown OnlineTempHeadLSTM mode={mode!r}")
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
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(3)
            ]
        )
        if self.mode == "moe":
            self.temp_gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 3))

    @staticmethod
    def _hard_head_indices(temp_scaled: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(temp_scaled[..., 0], dtype=torch.long)
        idx = torch.where(temp_scaled[..., 0] > -0.45, torch.ones_like(idx), idx)
        idx = torch.where(temp_scaled[..., 0] > 0.65, torch.full_like(idx, 2), idx)
        return idx

    def forward_sequence(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        z = self.input_proj(x)
        h, state = self.rnn(z, state)
        h = self.norm(h)
        logits = torch.stack([head(h) for head in self.heads], dim=-1).squeeze(-2)
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        if self.mode == "hard":
            idx = self._hard_head_indices(temp)
            selected = torch.gather(logits, dim=-1, index=idx.unsqueeze(-1))
            return torch.sigmoid(selected), state
        gate = torch.softmax(self.temp_gate(temp), dim=-1)
        return torch.sigmoid(torch.sum(logits * gate, dim=-1, keepdim=True)), state


class OnlineTempExpertLSTM(nn.Module):
    def __init__(self, input_dim: int, feature_cols: list[str], hidden_size: int, layers: int, dropout: float):
        super().__init__()
        if "T" not in feature_cols:
            raise RuntimeError(f"Temperature-expert online LSTM requires T, got {feature_cols}")
        self.temp_idx = int(feature_cols.index("T"))
        self.experts = nn.ModuleList(
            [
                OnlineLSTM(input_dim, hidden_size, layers, dropout)
                for _ in range(3)
            ]
        )

    def branch_index(self, x: torch.Tensor) -> int:
        temp = x[:, :, self.temp_idx:self.temp_idx + 1]
        idx = OnlineTempHeadLSTM._hard_head_indices(temp).reshape(-1)
        counts = torch.bincount(idx, minlength=3)
        return int(torch.argmax(counts).detach().cpu())

    def forward_sequence(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        branch = self.branch_index(x)
        return self.experts[branch].forward_sequence(x, state)


class OnlineContextLSTM(nn.Module):
    def __init__(self, input_dim: int, feature_cols: list[str], hidden_size: int, layers: int, dropout: float):
        super().__init__()
        context_cols = [c for c in ("V_corr_raw", "T") if c in feature_cols]
        if len(context_cols) != 2:
            raise RuntimeError(f"Context online LSTM requires V_corr_raw and T, got {feature_cols}")
        self.context_indices = [feature_cols.index(c) for c in context_cols]
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
            nn.Linear(hidden_size + len(self.context_indices), hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )

    def sequence_context(self, x_all: torch.Tensor) -> torch.Tensor:
        return x_all[:, 0:1, :].index_select(dim=2, index=torch.as_tensor(self.context_indices, device=x_all.device))

    def forward_sequence(
        self,
        x: torch.Tensor,
        state=None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if context is None:
            context = self.sequence_context(x)
        z = self.input_proj(x)
        h, state = self.rnn(z, state)
        ctx = context.expand(x.size(0), x.size(1), context.size(2))
        pred = torch.sigmoid(self.head(torch.cat([self.norm(h), ctx], dim=2)))
        return pred, state


class OnlineInitContextLSTM(nn.Module):
    def __init__(self, input_dim: int, feature_cols: list[str], hidden_size: int, layers: int, dropout: float):
        super().__init__()
        context_cols = [c for c in ("V_corr_raw", "T") if c in feature_cols]
        if len(context_cols) != 2:
            raise RuntimeError(f"Init-context online LSTM requires V_corr_raw and T, got {feature_cols}")
        self.context_indices = [feature_cols.index(c) for c in context_cols]
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.init_head = nn.Sequential(
            nn.Linear(len(self.context_indices), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * self.layers * self.hidden_size),
        )
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=float(dropout) if self.layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + len(self.context_indices), hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )

    def sequence_context(self, x_all: torch.Tensor) -> torch.Tensor:
        return x_all[:, 0:1, :].index_select(dim=2, index=torch.as_tensor(self.context_indices, device=x_all.device))

    def initial_state(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.size(0)
        init = torch.tanh(self.init_head(context[:, 0, :]))
        init = init.view(batch, 2, self.layers, self.hidden_size).permute(1, 2, 0, 3).contiguous()
        return init[0], init[1]

    def forward_sequence(
        self,
        x: torch.Tensor,
        state=None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if context is None:
            context = self.sequence_context(x)
        if state is None:
            state = self.initial_state(context)
        z = self.input_proj(x)
        h, state = self.rnn(z, state)
        ctx = context.expand(x.size(0), x.size(1), context.size(2))
        pred = torch.sigmoid(self.head(torch.cat([self.norm(h), ctx], dim=2)))
        return pred, state


class OnlineAnchorResidualLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        dropout: float,
        residual_limit: float,
    ):
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.feature_cols = list(feature_cols)
        anchor_cols = [c for c in ("V_corr_raw", "T") if c in self.feature_cols]
        if len(anchor_cols) != 2:
            raise RuntimeError(f"Anchor residual online LSTM requires V_corr_raw and T, got {self.feature_cols}")
        self.anchor_indices = [self.feature_cols.index(c) for c in anchor_cols]
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
        self.anchor_head = nn.Sequential(
            nn.Linear(len(self.anchor_indices), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size + len(self.anchor_indices), hidden_size),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, 1),
        )

    def forward_sequence(self, x: torch.Tensor, state=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        anchor_x = x.index_select(dim=2, index=torch.as_tensor(self.anchor_indices, device=x.device))
        anchor = torch.sigmoid(self.anchor_head(anchor_x))
        z = self.input_proj(x)
        h, state = self.rnn(z, state)
        h = self.norm(h)
        residual_input = torch.cat([h, anchor_x], dim=2)
        residual = float(self.residual_limit) * torch.tanh(self.residual_head(residual_input))
        return (anchor + residual).clamp(0.0, 1.0), state


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()


def make_model(cfg: OnlineG4LOPOConfig, feature_cols: list[str]) -> nn.Module:
    if cfg.model_kind == "plain":
        return OnlineLSTM(len(feature_cols), int(cfg.hidden_size), int(cfg.layers), float(cfg.dropout)).to(device)
    if cfg.model_kind == "temp_heads":
        return OnlineTempHeadLSTM(
            len(feature_cols),
            feature_cols,
            int(cfg.hidden_size),
            int(cfg.layers),
            float(cfg.dropout),
            mode="hard",
        ).to(device)
    if cfg.model_kind == "temp_moe":
        return OnlineTempHeadLSTM(
            len(feature_cols),
            feature_cols,
            int(cfg.hidden_size),
            int(cfg.layers),
            float(cfg.dropout),
            mode="moe",
        ).to(device)
    if cfg.model_kind == "temp_experts":
        return OnlineTempExpertLSTM(
            len(feature_cols),
            feature_cols,
            int(cfg.hidden_size),
            int(cfg.layers),
            float(cfg.dropout),
        ).to(device)
    if cfg.model_kind == "plain_context":
        return OnlineContextLSTM(
            len(feature_cols),
            feature_cols,
            int(cfg.hidden_size),
            int(cfg.layers),
            float(cfg.dropout),
        ).to(device)
    if cfg.model_kind == "init_context":
        return OnlineInitContextLSTM(
            len(feature_cols),
            feature_cols,
            int(cfg.hidden_size),
            int(cfg.layers),
            float(cfg.dropout),
        ).to(device)
    if cfg.model_kind == "anchor_residual":
        return OnlineAnchorResidualLSTM(
            len(feature_cols),
            feature_cols,
            int(cfg.hidden_size),
            int(cfg.layers),
            float(cfg.dropout),
            float(cfg.residual_limit),
        ).to(device)
    raise ValueError(f"Unknown model_kind={cfg.model_kind!r}")


def sequence_context(model: nn.Module, x_all: torch.Tensor):
    if hasattr(model, "sequence_context"):
        return model.sequence_context(x_all)
    return None


def forward_sequence(model: nn.Module, x: torch.Tensor, state, context):
    if context is not None:
        return model.forward_sequence(x, state, context)
    return model.forward_sequence(x, state)


def loss_values(pred: torch.Tensor, y: torch.Tensor, cfg: OnlineG4LOPOConfig) -> torch.Tensor:
    if cfg.loss_kind == "mae":
        return torch.abs(pred - y)
    if cfg.loss_kind == "mse":
        return (pred - y).square()
    if cfg.loss_kind == "huber":
        return F.smooth_l1_loss(pred, y, beta=float(cfg.huber_beta), reduction="none")
    raise ValueError(f"Unknown loss_kind={cfg.loss_kind!r}")


def temp_weight(temp: float, cfg: OnlineG4LOPOConfig) -> float:
    if np.isclose(float(temp), 0.0):
        return float(cfg.weight_0)
    if np.isclose(float(temp), 25.0):
        return float(cfg.weight_25)
    if np.isclose(float(temp), 45.0):
        return float(cfg.weight_45)
    return 1.0


def make_soc_weight_lookup(train_tensors: list[dict[str, object]], cfg: OnlineG4LOPOConfig) -> torch.Tensor | None:
    if cfg.soc_weight_mode == "none":
        return None
    if cfg.soc_weight_mode == "edge":
        return torch.as_tensor(
            [-1.0, float(cfg.soc_edge_low), float(cfg.soc_edge_high), float(cfg.soc_weight_strength)],
            dtype=torch.float32,
            device=device,
        )
    if cfg.soc_weight_mode != "uniform_bins":
        raise ValueError(f"Unknown soc_weight_mode={cfg.soc_weight_mode!r}")
    ys = []
    for item in train_tensors:
        y = item["y"]
        assert isinstance(y, torch.Tensor)
        ys.append(y.detach().flatten().cpu().numpy())
    all_y = np.concatenate(ys) if ys else np.empty(0, dtype=np.float32)
    n_bins = max(2, int(cfg.soc_weight_bins))
    counts, _ = np.histogram(np.clip(all_y, 0.0, 1.0), bins=n_bins, range=(0.0, 1.0))
    counts = np.maximum(counts.astype(np.float64), 1.0)
    inv = counts.mean() / counts
    inv = inv / max(float(inv.mean()), 1e-12)
    inv = np.clip(inv, 1.0 / max(float(cfg.soc_weight_clip), 1.0), max(float(cfg.soc_weight_clip), 1.0))
    strength = min(max(float(cfg.soc_weight_strength), 0.0), 1.0)
    weights = (1.0 - strength) + strength * inv
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.as_tensor(weights.astype(np.float32), device=device)


def soc_loss_weight(y: torch.Tensor, lookup: torch.Tensor | None) -> torch.Tensor | None:
    if lookup is None:
        return None
    if int(lookup.numel()) == 4 and float(lookup[0].detach().cpu()) < 0.0:
        low = float(lookup[1].detach().cpu())
        high = float(lookup[2].detach().cpu())
        strength = float(lookup[3].detach().cpu())
        edge = ((y.detach() <= low) | (y.detach() >= high)).to(dtype=y.dtype)
        weights = 1.0 + strength * edge
        return weights / weights.mean().clamp_min(1e-6)
    n_bins = int(lookup.numel())
    idx = torch.clamp((y.detach() * float(n_bins)).long(), min=0, max=n_bins - 1)
    return lookup.index_select(0, idx.reshape(-1)).reshape_as(y)


def tensorize_frames(frames: list[pd.DataFrame], feature_cols: list[str]) -> list[dict[str, object]]:
    out = []
    for frame in frames:
        f = frame.reset_index(drop=True)
        out.append(
            {
                "frame": f,
                "x": torch.as_tensor(
                    np.ascontiguousarray(f[feature_cols].to_numpy(np.float32))[None, :, :],
                    device=device,
                ),
                "y": torch.as_tensor(
                    np.ascontiguousarray(f["SOC_physical"].to_numpy(np.float32))[None, :, None],
                    device=device,
                ),
                "temperature": float(f["temperature"].iloc[0]),
            }
        )
    return out


def train_model(
    model: nn.Module,
    train_tensors: list[dict[str, object]],
    cfg: OnlineG4LOPOConfig,
    valid_tensors: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor] | None, int, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history = []
    best_state = None
    best_epoch = int(cfg.epochs)
    best_valid_mae = float("inf")
    chunk_len = max(1, int(cfg.chunk_len))
    order = list(range(len(train_tensors)))
    soc_weight_lookup = make_soc_weight_lookup(train_tensors, cfg)
    for ep in range(1, int(cfg.epochs) + 1):
        random.shuffle(order)
        model.train()
        losses = []
        maes = []
        smoothes = []
        monos = []
        for idx in order:
            item = train_tensors[idx]
            x_all = item["x"]
            y_all = item["y"]
            assert isinstance(x_all, torch.Tensor) and isinstance(y_all, torch.Tensor)
            weight = temp_weight(float(item["temperature"]), cfg)
            state = None
            context = sequence_context(model, x_all)
            for start in range(0, x_all.size(1), chunk_len):
                x = x_all[:, start : start + chunk_len, :]
                y = y_all[:, start : start + chunk_len, :]
                if x.numel() == 0:
                    continue
                pred, state = forward_sequence(model, x, state, context)
                raw_point_loss = loss_values(pred, y, cfg)
                y_weight = soc_loss_weight(y, soc_weight_lookup)
                point_loss = (raw_point_loss * y_weight).mean() if y_weight is not None else raw_point_loss.mean()
                loss = float(weight) * point_loss
                if float(cfg.lambda_smooth) > 0.0 and pred.size(1) > 1:
                    smooth = torch.abs((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])).mean()
                    loss = loss + float(cfg.lambda_smooth) * smooth
                else:
                    smooth = loss.new_tensor(0.0)
                if float(cfg.lambda_mono) > 0.0 and pred.size(1) > 1:
                    mono = F.relu(pred[:, 1:] - pred[:, :-1]).mean()
                    loss = loss + float(cfg.lambda_mono) * mono
                else:
                    mono = loss.new_tensor(0.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                state = detach_state(state)
                losses.append(float(loss.detach().cpu()))
                maes.append(float(torch.abs(pred.detach() - y).mean().cpu()))
                smoothes.append(float(smooth.detach().cpu()))
                monos.append(float(mono.detach().cpu()))
        row = {
            "epoch": int(ep),
            "loss": float(np.mean(losses)) if losses else np.nan,
            "mae_loss": float(np.mean(maes)) if maes else np.nan,
            "smooth_loss": float(np.mean(smoothes)) if smoothes else np.nan,
            "mono_loss": float(np.mean(monos)) if monos else np.nan,
        }
        should_report = ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0
        selector_metric = None
        if cfg.checkpoint_selector == "train_loss":
            selector_metric = float(row["loss"])
        elif cfg.checkpoint_selector == "train_mae":
            selector_metric = float(row["mae_loss"])
        if (
            cfg.checkpoint_selector in {"train_loss", "train_mae"}
            and np.isfinite(selector_metric)
            and selector_metric < best_valid_mae
        ):
            best_valid_mae = float(selector_metric)
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if valid_tensors and should_report and cfg.checkpoint_selector == "valid":
            valid_pred = predict_frames(model, valid_tensors, "valid", cfg, int(ep))
            valid_metrics = by_temperature(valid_pred)
            valid25 = valid_metrics[np.isclose(valid_metrics["temperature_C"].astype(float), 25.0)]
            if len(valid25):
                valid_mae = float(valid25["MAE_pct"].mean())
                row["valid25_MAE_pct"] = valid_mae
            elif len(valid_metrics):
                valid_mae = float(valid_metrics["MAE_pct"].mean())
                row["valid_MAE_pct"] = valid_mae
            else:
                valid_mae = float("inf")
            if np.isfinite(valid_mae) and valid_mae < best_valid_mae:
                best_valid_mae = float(valid_mae)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if should_report:
            valid_msg = ""
            if "valid25_MAE_pct" in row:
                valid_msg = f" valid25={row['valid25_MAE_pct']:.3f}%"
            elif "valid_MAE_pct" in row:
                valid_msg = f" valid={row['valid_MAE_pct']:.3f}%"
            print(
                f"{cfg.output_prefix} epoch={ep} loss={row['loss']:.5f} mae={row['mae_loss']:.5f}{valid_msg}",
                flush=True,
            )
    return pd.DataFrame(history), best_state, best_epoch, best_valid_mae


@torch.no_grad()
def predict_frames(
    model: nn.Module,
    tensors: list[dict[str, object]],
    split: str,
    cfg: OnlineG4LOPOConfig,
    epoch: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    chunk_len = max(1, int(cfg.eval_chunk_len))
    for item in tensors:
        f = item["frame"]
        x_all = item["x"]
        assert isinstance(f, pd.DataFrame) and isinstance(x_all, torch.Tensor)
        preds = []
        state = None
        context = sequence_context(model, x_all)
        for start in range(0, x_all.size(1), chunk_len):
            pred, state = forward_sequence(model, x_all[:, start : start + chunk_len, :], state, context)
            preds.append(pred.detach().cpu().numpy()[0, :, 0])
        y_pred = np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)
        out = pd.DataFrame(
            {
                "split": split,
                "seed": int(cfg.seed),
                "variant": f"online_g4_lstm_{cfg.model_kind}",
                "epoch": int(epoch),
                "trajectory_id": f["trajectory_id"].to_numpy(),
                "file_name": f["file_name"].to_numpy(),
                "drive_cycle": f["drive_cycle"].to_numpy(),
                "temperature": f["temperature"].to_numpy(),
                "temperature_C": f["temperature"].to_numpy(),
                "end_index": f["end_index"].to_numpy(),
                "y_true": f["SOC_physical"].to_numpy(np.float32),
                "y_pred": y_pred.astype(np.float32),
            }
        )
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def by_temperature(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for (split, seed, variant, epoch, temp), g in pred.groupby(
        ["split", "seed", "variant", "epoch", "temperature_C"],
        sort=True,
    ):
        err = g["error"].to_numpy(np.float64)
        rows.append(
            {
                "split": split,
                "seed": int(seed),
                "variant": variant,
                "epoch": int(epoch),
                "temperature_C": float(temp),
                "n_points": int(len(g)),
                "MAE_pct": float(g["abs_error"].mean() * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "bias_pct": float(g["error"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def write_test_summary(metrics: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    piv = test.pivot_table(
        index=["seed", "variant", "epoch"],
        columns="temperature_C",
        values="MAE_pct",
        aggfunc="mean",
    ).reset_index()
    for c in [0.0, 25.0, 45.0]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv["max_target"] = piv[[0.0, 25.0, 45.0]].max(axis=1)
    piv["target_met"] = (piv[0.0] < 1.0) & (piv[25.0] < 0.7) & (piv[45.0] < 0.3)
    piv.sort_values(["seed", "target_met", "max_target"], ascending=[True, False, True]).to_csv(out_path, index=False)
    return piv


def parse_profiles(raw: str) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text or text.upper() == "NONE":
        return ()
    return tuple(x.strip().upper() for x in text.split(",") if x.strip())


def run(cfg: OnlineG4LOPOConfig) -> dict[str, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    set_seed(int(cfg.seed))

    feature_cols = _selected_feature_columns(str(cfg.feature_set))
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")

    frame_cfg = TrainDSTSelectorConfig(
        base_dir=cfg.base_dir,
        raw_root=cfg.raw_root,
        output_prefix=cfg.output_prefix,
        seeds=(int(cfg.seed),),
        train_profiles=tuple(cfg.train_profiles),
        valid_profiles=tuple(cfg.valid_profiles),
        test_profiles=tuple(cfg.test_profiles),
        feature_set=str(cfg.feature_set),
        v_corr_tau_s=float(cfg.v_corr_tau_s),
        v_pol_mid_tau_s=float(cfg.v_pol_mid_tau_s),
        v_pol_slow_tau_s=float(cfg.v_pol_slow_tau_s),
        v_hys_tau_s=float(cfg.v_hys_tau_s),
    )
    r0_df = estimate_r0_by_temperature(files, tuple(cfg.train_profiles))
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(frame_cfg, files, r0_df))
    if not frames["train"] or not frames["test"]:
        raise RuntimeError("Expected non-empty train and test frame lists.")
    available = set().union(*(set(frame.columns) for split_frames in frames.values() for frame in split_frames))
    missing = [col for col in feature_cols if col not in available]
    if missing:
        raise RuntimeError(f"Selected feature_set={cfg.feature_set!r} has missing columns: {missing}")
    write_input_schema(feature_cols, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    write_leakage_audit(feature_cols, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")

    scaled, _ = make_scaled_frames_for_ablation(frames, feature_cols)
    train_tensors = tensorize_frames(scaled["train"], feature_cols)
    test_tensors = tensorize_frames(scaled["test"], feature_cols)
    valid_tensors = tensorize_frames(scaled["valid"], feature_cols) if scaled["valid"] else []

    model = make_model(cfg, feature_cols)
    history, best_state, best_epoch, best_valid_mae = train_model(model, train_tensors, cfg, valid_tensors)
    eval_epoch = int(cfg.epochs)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
        eval_epoch = int(best_epoch)
    pred_frames = [predict_frames(model, train_tensors, "train", cfg, eval_epoch)]
    if valid_tensors:
        pred_frames.append(predict_frames(model, valid_tensors, "valid", cfg, eval_epoch))
    pred_frames.append(predict_frames(model, test_tensors, "test", cfg, eval_epoch))
    pred = pd.concat(pred_frames, ignore_index=True)
    metrics = by_temperature(pred)
    summary = write_test_summary(metrics, out_dir / f"{cfg.output_prefix}_test_summary.csv")

    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    metrics.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    if bool(cfg.save_predictions):
        pred.to_csv(out_dir / f"{cfg.output_prefix}_prediction_rows.csv.gz", index=False, compression="gzip")
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": feature_cols,
        "input_feature_dim": len(feature_cols),
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "online_causal_streaming": True,
        "state_carried_between_steps": "LSTM hidden/cell state only",
        "state_reset": "per trajectory",
        "selected_epoch": eval_epoch,
        "validation_selector": (
            "valid 25C MAE at print/eval checkpoints"
            if cfg.checkpoint_selector == "valid" and valid_tensors
            else str(cfg.checkpoint_selector)
        ),
        "best_valid_MAE_pct": None if not np.isfinite(best_valid_mae) else float(best_valid_mae),
        "train_trajectories": len(scaled["train"]),
        "valid_trajectories": len(scaled["valid"]),
        "test_trajectories": len(scaled["test"]),
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("Online G4 LSTM test summary:")
    print(summary.to_string(index=False), flush=True)
    return {"history": history, "metrics": metrics, "summary": summary, "pred": pred}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online full-profile G4 LSTM LOPO screen.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=OnlineG4LOPOConfig.raw_root)
    p.add_argument("--output-prefix", default=OnlineG4LOPOConfig.output_prefix)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-profiles", default="VALIDATION,DST,FUDS")
    p.add_argument("--valid-profiles", default="NONE")
    p.add_argument("--test-profiles", default="US06")
    p.add_argument("--feature-set", default="paper_g4_all_ema")
    p.add_argument(
        "--model-kind",
        choices=[
            "plain",
            "temp_heads",
            "temp_moe",
            "temp_experts",
            "plain_context",
            "init_context",
            "anchor_residual",
        ],
        default="plain",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--eval-chunk-len", type=int, default=1024)
    p.add_argument("--loss-kind", choices=["huber", "mae", "mse"], default="huber")
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--weight-0", type=float, default=0.8)
    p.add_argument("--weight-25", type=float, default=2.2)
    p.add_argument("--weight-45", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--lambda-mono", type=float, default=0.0)
    p.add_argument("--checkpoint-selector", choices=["final", "valid", "train_loss", "train_mae"], default="final")
    p.add_argument("--soc-weight-mode", choices=["none", "uniform_bins", "edge"], default="none")
    p.add_argument("--soc-weight-strength", type=float, default=1.0)
    p.add_argument("--soc-weight-bins", type=int, default=10)
    p.add_argument("--soc-weight-clip", type=float, default=4.0)
    p.add_argument("--soc-edge-low", type=float, default=0.12)
    p.add_argument("--soc-edge-high", type=float, default=0.72)
    p.add_argument("--residual-limit", type=float, default=0.25)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        OnlineG4LOPOConfig(
            base_dir=Path(args.base_dir),
            raw_root=Path(args.raw_root),
            output_prefix=str(args.output_prefix),
            seed=int(args.seed),
            train_profiles=parse_profiles(args.train_profiles),
            valid_profiles=parse_profiles(args.valid_profiles),
            test_profiles=parse_profiles(args.test_profiles),
            feature_set=str(args.feature_set),
            model_kind=str(args.model_kind),
            epochs=int(args.epochs),
            hidden_size=int(args.hidden_size),
            layers=int(args.layers),
            dropout=float(args.dropout),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            chunk_len=int(args.chunk_len),
            eval_chunk_len=int(args.eval_chunk_len),
            loss_kind=str(args.loss_kind),
            huber_beta=float(args.huber_beta),
            weight_0=float(args.weight_0),
            weight_25=float(args.weight_25),
            weight_45=float(args.weight_45),
            lambda_smooth=float(args.lambda_smooth),
            lambda_mono=float(args.lambda_mono),
            checkpoint_selector=str(args.checkpoint_selector),
            soc_weight_mode=str(args.soc_weight_mode),
            soc_weight_strength=float(args.soc_weight_strength),
            soc_weight_bins=int(args.soc_weight_bins),
            soc_weight_clip=float(args.soc_weight_clip),
            soc_edge_low=float(args.soc_edge_low),
            soc_edge_high=float(args.soc_edge_high),
            residual_limit=float(args.residual_limit),
            print_every=int(args.print_every),
            save_predictions=bool(args.save_predictions),
        )
    )


if __name__ == "__main__":
    main()
