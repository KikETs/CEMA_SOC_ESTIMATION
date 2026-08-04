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
from .deep_no_leak_experiment import CausalConvBlock, SequenceWindowDataset, make_eval_loader
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
from .nmc_vcorr_it_goal_remote_screen import Variant, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS, attach_eval_features, make_endpoint_lookup
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


BASE_PREFIX = "nmc_goal_vcorr_it_h64_w50_single_checkpoint_moe_seed0"


@dataclass
class SingleCheckpointMoEConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    window_len: int = 50
    stride: int = 3
    epochs: int = 180
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    num_workers: int = 4
    prefetch_factor: int = 4
    print_every: int = 20
    valid_every: int = 5
    variant_set: str = "screen"
    low_current_threshold_A: float = 0.05
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


@dataclass
class SingleCheckpointMoEVariant:
    name: str
    gate_mode: str = "hard_temp"
    expert_0: str = "tcn6"
    expert_25: str = "tcn5"
    expert_45: str = "lstm1"
    dropout: float = 0.06
    kernel_size: int = 5
    norm_kind: str = "channel"
    lambda_rex: float = 2.0
    lr: float = 8e-4
    weight_0: float = 2.0
    weight_25: float = 3.0
    weight_45: float = 3.0
    seq_weight: float = 0.0
    gate_entropy: float = 0.0
    selection: str = "tri45_goal"


DESIGN_INTENT = [
    {
        "component": "single checkpoint MoE",
        "why": "All experts live in one model state dict and are trained in one run; inference does not load or swap temperature-specific checkpoints.",
    },
    {
        "component": "temperature-conditioned gate",
        "why": "Temperature is known online and changes the voltage-response map; the gate uses T only, not SOC labels or cumulative charge.",
    },
    {
        "component": "heterogeneous experts",
        "why": "Earlier valid screens showed TCN branches help 0C/25C while a short LSTM branch is better for 45C relaxation-like behavior.",
    },
    {
        "component": "validation-only selection",
        "why": "The selected epoch is the minimum VALIDATION validation tri45 score; FUDS test is evaluated only after selection.",
    },
]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class TCNExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, layers: int, kernel_size: int, dropout: float, norm_kind: str):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.blocks = nn.Sequential(
            *[
                CausalConvBlock(
                    hidden_size,
                    kernel_size=int(kernel_size),
                    dilation=2 ** i,
                    dropout=float(dropout),
                    norm_kind=str(norm_kind),
                )
                for i in range(int(layers))
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).transpose(1, 2)
        h = self.blocks(h).transpose(1, 2)
        return self.head(self.norm(h))


class LSTMExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=int(layers),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h, _state = self.rnn(self.input_proj(x))
        return self.head(self.norm(h))


def make_expert(kind: str, input_dim: int, hidden_size: int, variant: SingleCheckpointMoEVariant) -> nn.Module:
    kind = str(kind).lower()
    if kind == "tcn4":
        return TCNExpert(input_dim, hidden_size, 4, variant.kernel_size, variant.dropout, variant.norm_kind)
    if kind == "tcn5":
        return TCNExpert(input_dim, hidden_size, 5, variant.kernel_size, variant.dropout, variant.norm_kind)
    if kind == "tcn6":
        return TCNExpert(input_dim, hidden_size, 6, variant.kernel_size, variant.dropout, variant.norm_kind)
    if kind == "lstm1":
        return LSTMExpert(input_dim, hidden_size, 1, variant.dropout)
    if kind == "lstm2":
        return LSTMExpert(input_dim, hidden_size, 2, variant.dropout)
    raise ValueError(f"Unknown expert kind={kind}")


class SingleCheckpointMoE(nn.Module):
    def __init__(self, variant: SingleCheckpointMoEVariant, input_dim: int = 3, hidden_size: int = 64):
        super().__init__()
        self.variant = variant
        self.temp_idx = 2
        self.experts = nn.ModuleList(
            [
                make_expert(variant.expert_0, input_dim, hidden_size, variant),
                make_expert(variant.expert_25, input_dim, hidden_size, variant),
                make_expert(variant.expert_45, input_dim, hidden_size, variant),
            ]
        )
        if variant.gate_mode == "soft_temp":
            self.gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 3))
        elif variant.gate_mode != "hard_temp":
            raise ValueError(f"Unknown gate_mode={variant.gate_mode}")

    def hard_temp_indices(self, temp_scaled: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(temp_scaled[..., 0], dtype=torch.long)
        idx = torch.where(temp_scaled[..., 0] > -0.45, torch.ones_like(idx), idx)
        idx = torch.where(temp_scaled[..., 0] > 0.65, torch.full_like(idx, 2), idx)
        return idx

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        temp = x[..., self.temp_idx:self.temp_idx + 1]
        if self.variant.gate_mode == "soft_temp":
            return torch.softmax(self.gate(temp), dim=-1)
        idx = self.hard_temp_indices(temp)
        return F.one_hot(idx, num_classes=3).to(dtype=x.dtype)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([expert.forward_logits(x) for expert in self.experts], dim=-1).squeeze(-2)
        weights = self.gate_weights(x)
        return torch.sum(logits * weights, dim=-1, keepdim=True)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]

    def gate_entropy_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant.gate_mode != "soft_temp":
            return x.new_tensor(0.0)
        w = self.gate_weights(x)
        entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1).mean()
        return entropy


def variant_list(variant_set: str) -> list[SingleCheckpointMoEVariant]:
    if variant_set == "fast":
        return [
            SingleCheckpointMoEVariant(
                "hard_tcn6_tcn5_lstm1_w25x3_w45x3_seq0",
                gate_mode="hard_temp",
                expert_0="tcn6",
                expert_25="tcn5",
                expert_45="lstm1",
                weight_0=2.0,
                weight_25=3.0,
                weight_45=3.0,
                seq_weight=0.0,
            ),
            SingleCheckpointMoEVariant(
                "hard_tcn6_tcn5_lstm1_w25x4_w45x4_seq0p2",
                gate_mode="hard_temp",
                expert_0="tcn6",
                expert_25="tcn5",
                expert_45="lstm1",
                weight_0=2.0,
                weight_25=4.0,
                weight_45=4.0,
                seq_weight=0.2,
            ),
        ]
    return [
        SingleCheckpointMoEVariant(
            "hard_tcn6_tcn5_lstm1_w25x3_w45x3_seq0",
            gate_mode="hard_temp",
            expert_0="tcn6",
            expert_25="tcn5",
            expert_45="lstm1",
            weight_0=2.0,
            weight_25=3.0,
            weight_45=3.0,
            seq_weight=0.0,
        ),
        SingleCheckpointMoEVariant(
            "hard_tcn6_tcn5_lstm1_w25x4_w45x4_seq0p2",
            gate_mode="hard_temp",
            expert_0="tcn6",
            expert_25="tcn5",
            expert_45="lstm1",
            weight_0=2.0,
            weight_25=4.0,
            weight_45=4.0,
            seq_weight=0.2,
        ),
        SingleCheckpointMoEVariant(
            "hard_tcn6_tcn5_lstm2_w25x4_w45x4_seq0p5",
            gate_mode="hard_temp",
            expert_0="tcn6",
            expert_25="tcn5",
            expert_45="lstm2",
            weight_0=2.0,
            weight_25=4.0,
            weight_45=4.0,
            seq_weight=0.5,
        ),
        SingleCheckpointMoEVariant(
            "soft_tcn6_tcn5_lstm1_w25x4_w45x4_seq0p2_ent0p01",
            gate_mode="soft_temp",
            expert_0="tcn6",
            expert_25="tcn5",
            expert_45="lstm1",
            weight_0=2.0,
            weight_25=4.0,
            weight_45=4.0,
            seq_weight=0.2,
            gate_entropy=-0.01,
        ),
    ]


def tri45_score(overall: pd.DataFrame, by_temp: pd.DataFrame) -> float:
    if overall.empty or by_temp.empty:
        return float("inf")
    overall_mae = float(overall["MAE_pct"].iloc[0])

    def t_mae(temp: float) -> float:
        sub = by_temp[np.isclose(by_temp["temperature_C"], temp)]
        return float(sub["MAE_pct"].iloc[0]) if len(sub) else overall_mae

    mae0 = t_mae(0.0)
    mae25 = t_mae(25.0)
    mae45 = t_mae(45.0)
    target0, target25, target45 = 1.0, 0.7, 0.3
    excess = max(0.0, mae0 - target0) + 1.5 * max(0.0, mae25 - target25) + 2.0 * max(0.0, mae45 - target45)
    normalized_worst = max(mae0 / target0, mae25 / target25, mae45 / target45)
    return float(normalized_worst + 3.0 * excess + 0.05 * overall_mae)


def add_goal_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = False
    if out.empty or "variant" not in out.columns:
        return out
    for variant, idx in out.groupby("variant").groups.items():
        sub = out.loc[idx]
        vals = {float(r.temperature_C): float(r.MAE_pct) for r in sub.itertuples() if hasattr(r, "temperature_C")}
        ok = vals.get(0.0, 999.0) < 1.0 and vals.get(25.0, 999.0) < 0.7 and vals.get(45.0, 999.0) < 0.3
        out.loc[idx, "goal_0C_lt1_25C_lt0p7_45C_lt0p3"] = bool(ok)
    return out


def predict_loader(model: nn.Module, loader, model_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    gate_rows = []
    with torch.no_grad():
        for x, y, meta in loader:
            x_dev = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred = model(x_dev)
            gates = model.gate_weights(x_dev)[:, -1, :].detach().cpu().numpy()
            mdf = collate_meta_to_frame(meta)
            mdf["model_name"] = model_name
            mdf["target_label"] = "physical"
            mdf["y_true"] = y.numpy()[:, -1, 0] if y.ndim == 3 else y.numpy()[:, 0]
            mdf["y_pred"] = pred.detach().cpu().numpy()[:, 0]
            mdf["gate_0C_expert"] = gates[:, 0]
            mdf["gate_25C_expert"] = gates[:, 1]
            mdf["gate_45C_expert"] = gates[:, 2]
            rows.append(mdf)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out["error"] = out["y_pred"] - out["y_true"]
        out["abs_error"] = np.abs(out["error"])
        if "temperature" in out.columns and "temperature_C" not in out.columns:
            out["temperature_C"] = out["temperature"].astype(float)
    return out


def evaluate(model: nn.Module, loader, model_name: str, lookup) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = attach_eval_features(predict_loader(model, loader, model_name), lookup)
    return pred, _overall_metrics(pred), variance_by_temperature(pred)


def train_variant(cfg: SingleCheckpointMoEConfig, variant: SingleCheckpointMoEVariant, frames, out_dir: Path):
    model_name = f"{cfg.output_prefix}_{variant.name}"
    scaled, _ = make_scaled_frames_for_ablation(frames, FEATURE_COLS)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    train_ds = SequenceWindowDataset(scaled["train"], FEATURE_COLS, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled["valid"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled["test"], FEATURE_COLS, cfg.window_len, 1, target_label="physical")
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    lookup = make_endpoint_lookup(frames, FEATURE_COLS)
    model = SingleCheckpointMoE(variant, input_dim=len(FEATURE_COLS), hidden_size=int(cfg.hidden_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(variant.lr), weight_decay=float(cfg.weight_decay))
    temp_variant = Variant(
        variant.name,
        recurrent="moe",
        layers=1,
        head_kind="linear",
        temp_mode="moe",
        dropout=variant.dropout,
        lambda_rex=variant.lambda_rex,
        lr=variant.lr,
        weight_0=variant.weight_0,
        weight_25=variant.weight_25,
        weight_45=variant.weight_45,
    )
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        by_group: dict[str, list[float]] = {}
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred_seq = model.forward_sequence(x)
            endpoint_loss = F.smooth_l1_loss(pred_seq[:, -1, :], y[:, -1, :], beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            if float(variant.seq_weight) > 0.0:
                seq_loss = F.smooth_l1_loss(pred_seq, y, beta=float(cfg.huber_beta), reduction="none").mean(dim=(1, 2))
                sample_loss = endpoint_loss + float(variant.seq_weight) * seq_loss
            else:
                sample_loss = endpoint_loss
            sw = temp_weights(meta, temp_variant, int(sample_loss.numel()))
            keys = group_keys(meta, cfg.rex_group)
            group_losses = []
            group_weights = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                g_loss = sample_loss.index_select(0, idx).mean()
                group_losses.append(g_loss)
                group_weights.append(sw.index_select(0, idx).mean())
                by_group.setdefault(key, []).append(float(g_loss.detach().cpu()))
            stack = torch.stack(group_losses)
            wstack = torch.stack(group_weights)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            loss = mean_loss + float(variant.lambda_rex) * rex_var
            if float(variant.gate_entropy) != 0.0:
                loss = loss + float(variant.gate_entropy) * model.gate_entropy_loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {
            "model_name": model_name,
            "variant": variant.name,
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_group_loss": float(mean_loss.detach().cpu()),
            "rex_var": float(rex_var.detach().cpu()),
        }
        for key, vals in by_group.items():
            row[f"train_loss_group_{key.replace('.', 'p').replace('-', 'N')}"] = float(np.mean(vals))
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.valid_every)) == 0:
            valid_pred, valid_overall, valid_by_temp = evaluate(model, valid_loader, model_name, lookup)
            score = tri45_score(valid_overall, valid_by_temp)
            row["valid_score"] = float(score)
            row["valid_MAE_pct"] = float(valid_overall["MAE_pct"].iloc[0]) if len(valid_overall) else float("nan")
            for temp in (0.0, 25.0, 45.0):
                sub = valid_by_temp[np.isclose(valid_by_temp["temperature_C"], temp)]
                row[f"valid_T{int(temp)}_MAE_pct"] = float(sub["MAE_pct"].iloc[0]) if len(sub) else float("nan")
            if np.isfinite(score) and score < best_score:
                best_score = float(score)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if ep == 1 or ep == int(cfg.epochs) or ep % max(1, int(cfg.print_every)) == 0:
            valid_msg = f" valid_score={row['valid_score']:.3f} valid_mae={row['valid_MAE_pct']:.3f}%" if "valid_score" in row else ""
            print(f"{model_name} epoch={ep} loss={row['loss']:.5f}{valid_msg}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        best_epoch = int(cfg.epochs)
    history_df = pd.DataFrame(history)
    pred = attach_eval_features(predict_loader(model, test_loader, model_name), lookup)
    valid_pred = attach_eval_features(predict_loader(model, valid_loader, model_name), lookup)
    for df in (pred, valid_pred):
        df["seed"] = int(cfg.seed)
        df["variant"] = variant.name
        df["selected_epoch"] = int(best_epoch)
        df["selection"] = variant.selection
        df["gate_mode"] = variant.gate_mode
        df["input_feature_dim"] = len(FEATURE_COLS)
    overall = _overall_metrics(pred)
    by_temp = add_goal_flags(variance_by_temperature(pred))
    by_traj = metrics_by_trajectory(pred)
    focus = focus_metrics(pred, cfg, model_name)
    valid_overall = _overall_metrics(valid_pred)
    valid_by_temp = add_goal_flags(variance_by_temperature(valid_pred))
    for df in (overall, by_temp, by_traj, focus, valid_overall, valid_by_temp):
        if not df.empty:
            df["variant"] = variant.name
            df["selected_epoch"] = int(best_epoch)
            df["selection"] = variant.selection
            df["gate_mode"] = variant.gate_mode
            df["input_feature_dim"] = len(FEATURE_COLS)
    history_df.to_csv(out_dir / f"{model_name}_history.csv", index=False)
    pred.to_csv(out_dir / f"{model_name}_prediction_rows.csv.gz", index=False, compression="gzip")
    valid_pred.to_csv(out_dir / f"{model_name}_valid_prediction_rows.csv.gz", index=False, compression="gzip")
    overall.to_csv(out_dir / f"{model_name}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{model_name}_by_temperature.csv", index=False)
    by_traj.to_csv(out_dir / f"{model_name}_by_trajectory.csv", index=False)
    focus.to_csv(out_dir / f"{model_name}_focus.csv", index=False)
    valid_overall.to_csv(out_dir / f"{model_name}_valid_overall.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{model_name}_valid_by_temperature.csv", index=False)
    print("Overall:")
    print(overall.to_string(index=False), flush=True)
    print("By temperature:")
    print(by_temp.to_string(index=False), flush=True)
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


def write_selection_audit(out_dir: Path, prefix: str, history: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in history.groupby("variant"):
        hg = g.dropna(subset=["valid_score"]).copy()
        selected = int(overall.loc[overall["variant"].eq(variant), "selected_epoch"].iloc[0])
        idx = hg["valid_score"].astype(float).idxmin()
        best_epoch = int(hg.loc[idx, "epoch"])
        rows.append(
            {
                "variant": variant,
                "selected_epoch": selected,
                "best_validation_epoch_from_history": best_epoch,
                "best_validation_score": float(hg.loc[idx, "valid_score"]),
                "status": "PASS" if selected == best_epoch else "FAIL",
                "detail": "selected epoch equals minimum validation score epoch; test metrics not used",
            }
        )
    audit = pd.DataFrame(rows).sort_values("best_validation_score")
    audit.to_csv(out_dir / f"{prefix}_selection_audit.csv", index=False)
    return audit


def write_report(cfg, out_dir, r0_df, schema, leakage, selection_audit, overall, by_temp, valid_by_temp):
    intent_df = pd.DataFrame(DESIGN_INTENT)
    pvt = by_temp.pivot_table(index=["variant", "selected_epoch", "gate_mode"], columns="temperature_C", values="MAE_pct").reset_index()
    pvt = pvt.rename(columns={0.0: "MAE_0C", 25.0: "MAE_25C", 45.0: "MAE_45C"})
    if len(pvt):
        pvt["goal"] = (pvt["MAE_0C"] < 1.0) & (pvt["MAE_25C"] < 0.7) & (pvt["MAE_45C"] < 0.3)
        pvt["score"] = np.maximum.reduce([pvt["MAE_0C"] / 1.0, pvt["MAE_25C"] / 0.7, pvt["MAE_45C"] / 0.3])
        pvt = pvt.sort_values(["goal", "score"], ascending=[False, True])
    lines = [
        "# NMC Vcorr/I/T Single-Checkpoint MoE Tri45 Experiment",
        "",
        "## Fixed Conditions",
        "- One saved model/checkpoint per candidate; no external temperature-specific checkpoint switching.",
        "- Inputs: V_corr_raw, I_raw, T only.",
        "- Window length: 50; hidden/model dimension: 64.",
        "- Train: DST + US06. Validation: VALIDATION. Test: FUDS.",
        "- Checkpoint selection: minimum validation tri45 score only.",
        "- Targets: 0C MAE < 1.0%p, 25C MAE < 0.7%p, 45C MAE < 0.3%p.",
        "",
        "## Design Intent",
        table_md(intent_df, ["component", "why"]),
        "",
        "## Candidate Summary",
        table_md(pvt, ["variant", "selected_epoch", "gate_mode", "MAE_0C", "MAE_25C", "MAE_45C", "goal", "score"]),
        "",
        "## Overall",
        table_md(overall, ["variant", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch", "selection", "gate_mode"]),
        "",
        "## By Temperature",
        table_md(by_temp, ["variant", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "jitter_ratio", "selected_epoch", "gate_mode", "goal_0C_lt1_25C_lt0p7_45C_lt0p3"]),
        "",
        "## Valid By Temperature",
        table_md(valid_by_temp, ["variant", "temperature_C", "n_windows", "MAE_pct", "RMSE_pct", "selected_epoch", "gate_mode"]),
        "",
        "## Selection Audit",
        table_md(selection_audit, ["variant", "selected_epoch", "best_validation_epoch_from_history", "best_validation_score", "status", "detail"]),
        "",
        "## Leakage Audit",
        table_md(leakage, ["audit_item", "status", "detail"]),
        "",
        "## Input Schema",
        table_md(schema, ["index_1based", "feature_name", "source"]),
        "",
        "## R0 / Vcorr Preprocessing",
        table_md(r0_df, ["temperature_C", "r0_ohm", "n_events", "r0_p20_ohm", "r0_p80_ohm"]),
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: SingleCheckpointMoEConfig):
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    if int(cfg.hidden_size) != 64 or int(cfg.window_len) != 50:
        raise ValueError("This experiment is fixed to hidden_size=64 and window_len=50.")
    if FEATURE_COLS != ["V_corr_raw", "I_raw", "T"]:
        raise ValueError(f"Expected FEATURE_COLS V_corr_raw/I_raw/T, got {FEATURE_COLS}")
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_single_checkpoint_moe_results"
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
    variants = variant_list(cfg.variant_set)
    results = []
    for variant in variants:
        set_seed(cfg.seed)
        print(f"===== single-checkpoint MoE {variant.name} =====", flush=True)
        results.append(train_variant(cfg, variant, frames, out_dir))
    overall = pd.concat([r["overall"] for r in results], ignore_index=True).sort_values("MAE_pct")
    by_temp = pd.concat([r["by_temperature"] for r in results], ignore_index=True)
    focus = pd.concat([r["focus"] for r in results], ignore_index=True)
    valid_overall = pd.concat([r["valid_overall"] for r in results], ignore_index=True).sort_values("MAE_pct")
    valid_by_temp = pd.concat([r["valid_by_temperature"] for r in results], ignore_index=True)
    history = pd.concat([r["history"] for r in results], ignore_index=True)
    overall.to_csv(out_dir / f"{cfg.output_prefix}_overall.csv", index=False)
    by_temp.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    valid_overall.to_csv(out_dir / f"{cfg.output_prefix}_valid_overall.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{cfg.output_prefix}_valid_by_temperature.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    selection_audit = write_selection_audit(out_dir, cfg.output_prefix, history, overall)
    metadata = {
        **asdict(cfg),
        "base_dir": str(cfg.base_dir),
        "raw_root": str(cfg.raw_root),
        "feature_columns": FEATURE_COLS,
        "input_feature_dim": len(FEATURE_COLS),
        "hidden_size_verified": 64,
        "window_len_verified": 50,
        "checkpoint_selection": "validation_only",
        "temperature_checkpoint_switching": False,
        "architecture": "single checkpoint internal temperature-conditioned MoE",
        "design_intent": DESIGN_INTENT,
        "variants": [asdict(v) for v in variants],
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "uses_absolute_time_or_progress": False,
        "uses_amp": False,
    }
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(cfg, out_dir, r0_df, schema, leakage, selection_audit, overall, by_temp, valid_by_temp)
    print("Single-checkpoint MoE overall:")
    print(overall.to_string(index=False), flush=True)
    print("Single-checkpoint MoE by temperature:")
    print(by_temp.to_string(index=False), flush=True)
    print(f"Report: {out_dir / (cfg.output_prefix + '_report.md')}", flush=True)
    return {
        "overall": overall,
        "by_temperature": by_temp,
        "focus": focus,
        "valid_overall": valid_overall,
        "valid_by_temperature": valid_by_temp,
        "history": history,
        "selection_audit": selection_audit,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC Vcorr/I/T single-checkpoint MoE tri45 experiment.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=SingleCheckpointMoEConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--valid-every", type=int, default=5)
    p.add_argument("--variant-set", default="screen", choices=["screen", "fast"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SingleCheckpointMoEConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
        valid_every=int(args.valid_every),
        variant_set=str(args.variant_set),
    )
    run(cfg)


if __name__ == "__main__":
    main()
