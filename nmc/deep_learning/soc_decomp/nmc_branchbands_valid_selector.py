from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import make_cfg
from .deep_no_leak_experiment import (
    AugmentedSequenceWindowDataset,
    AugmentedWindowDataset,
    DeepNoLeakTCN,
    DeepNoLeakTempAffineTCN,
    augmented_input_dim,
    feature_columns,
    make_eval_loader,
    predict,
    temp_balanced_rex_loss,
)
from .extrapolation_robustness import temperature_balanced_loader
from .nmc_branchbands_experiment import (
    NMCBranchBandsConfig,
    add_extra_prediction_features,
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
    focus_metrics,
    make_endpoint_feature_lookup,
    metrics_by_trajectory,
    write_input_schema,
    write_leakage_audit,
    write_start_audit,
)
from .runtime import configure_torch_runtime, device
from .training import attach_prediction_features, build_prediction_feature_lookup, make_scaled_frames_for_ablation
from .variance_control import _overall_metrics, variance_by_temperature


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def _valid_temp_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temp, g in pred.groupby("temperature"):
        rows.append(
            {
                "temperature_C": float(temp),
                "n_windows": int(len(g)),
                "MAE_pct": float(g["abs_error"].mean() * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(g["error"].to_numpy(np.float64) ** 2)) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("temperature_C").reset_index(drop=True)


def _selector_score(temp_table: pd.DataFrame, selector: str) -> float:
    vals = {
        float(r["temperature_C"]): float(r["MAE_pct"])
        for _, r in temp_table.iterrows()
        if np.isfinite(float(r["MAE_pct"]))
    }
    targets = [vals.get(t, np.inf) for t in (0.0, 25.0, 45.0)]
    if selector == "valid_worst_temp":
        return float(np.max(targets))
    if selector == "valid_mean_temp":
        return float(np.mean(targets))
    if selector == "valid_25":
        return float(vals.get(25.0, np.inf))
    raise ValueError(f"Unknown selector={selector!r}")


def _attach_for_metrics(pred: pd.DataFrame, frames, feature_cols: list[str], split: str, model_name: str) -> pd.DataFrame:
    generic_lookup = build_prediction_feature_lookup(frames)
    out = attach_prediction_features(
        pred.assign(split=split, ablation=model_name),
        generic_lookup,
        ablation_name=model_name,
        target_label="physical",
    )
    endpoint_lookup = make_endpoint_feature_lookup(frames, feature_cols)
    return add_extra_prediction_features(out, endpoint_lookup, feature_cols)


def _selected_feature_columns(name: str) -> list[str]:
    if str(name) == "vcorr_it":
        return ["V_corr_raw", "I_raw", "T"]
    return feature_columns(str(name))


def _augmented_indices(feature_cols: list[str], names: list[str], mode: str) -> list[int]:
    raw = [feature_cols.index(name) for name in names if name in feature_cols]
    if str(mode) == "raw":
        return raw
    n = len(feature_cols)
    out = list(raw)
    if str(mode) in {"delta_start", "delta_start_time", "delta_start_time_local_residual"}:
        out.extend([n + idx for idx in raw])
    if str(mode) in {"delta_start_time", "delta_start_time_local_residual"}:
        out.append(2 * n)
    if str(mode) == "delta_start_time_local_residual":
        out.extend(range(2 * n + 1, 2 * n + 8))
    return out


def _tcn_logits(model: DeepNoLeakTCN, x: torch.Tensor) -> torch.Tensor:
    hidden = model.encode_sequence(x)
    return model.head(hidden).transpose(1, 2)


class ObservabilityFusionTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        window_feature_mode: str,
        hidden_size: int,
        layers: int,
        kernel_size: int,
        norm_kind: str,
        dropout: float,
        vit_hidden: int = 64,
        vit_layers: int = 5,
        gate_prior_vit: float = 0.8,
        fusion_mode: str = "mixture",
        corr_logit_limit: float = 0.7,
    ):
        super().__init__()
        self.feature_cols = list(feature_cols)
        self.window_feature_mode = str(window_feature_mode)
        self.fusion_mode = str(fusion_mode)
        self.corr_logit_limit = float(corr_logit_limit)
        vit_names = ["V_corr_raw", "I_raw", "T"]
        gate_names = [
            "V_corr_raw",
            "I_raw",
            "T",
            "absI",
            "dI",
            "V_pol_raw",
            "V_pol_fast_raw",
            "V_pol_mid_raw",
            "V_pol_slow_raw",
            "V_residual_low",
            "V_residual_mid",
            "V_residual_high",
            "R0",
        ]
        self.vit_indices = _augmented_indices(self.feature_cols, vit_names, self.window_feature_mode)
        self.gate_indices = _augmented_indices(self.feature_cols, gate_names, self.window_feature_mode)
        if not self.vit_indices:
            raise ValueError("ObservabilityFusionTCN requires V_corr_raw/I_raw/T indices.")
        if not self.gate_indices:
            raise ValueError("ObservabilityFusionTCN requires label-free gate feature indices.")

        self.vit_branch = DeepNoLeakTCN(
            input_dim=len(self.vit_indices),
            hidden_size=int(vit_hidden),
            layers=int(vit_layers),
            kernel_size=int(kernel_size),
            norm_kind=str(norm_kind),
            dropout=float(dropout),
        )
        self.band_branch = DeepNoLeakTCN(
            input_dim=int(input_dim),
            hidden_size=int(hidden_size),
            layers=int(layers),
            kernel_size=int(kernel_size),
            norm_kind=str(norm_kind),
            dropout=float(dropout),
        )
        gate_dim = len(self.gate_indices) * 4 + 3
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 2),
        )
        p = float(np.clip(gate_prior_vit, 1e-3, 1.0 - 1e-3))
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([np.log(p), np.log(1.0 - p)], dtype=torch.float32))

    def _window_stats(self, x: torch.Tensor) -> torch.Tensor:
        xg = x[..., self.gate_indices]
        x_end = xg[:, -1, :]
        x_mean = xg.mean(dim=1)
        x_std = xg.std(dim=1, unbiased=False)
        x_delta = x_end - xg[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def gate_weights(self, x: torch.Tensor, vit_logits: torch.Tensor, band_logits: torch.Tensor) -> torch.Tensor:
        vit_end = vit_logits[:, -1, :]
        band_end = band_logits[:, -1, :]
        gate_input = torch.cat([self._window_stats(x), vit_end, band_end, vit_end - band_end], dim=1)
        return torch.softmax(self.gate(gate_input), dim=1)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        vit_x = x[..., self.vit_indices]
        vit_logits = _tcn_logits(self.vit_branch, vit_x)
        band_logits = _tcn_logits(self.band_branch, x)
        weights = self.gate_weights(x, vit_logits, band_logits)
        self.last_vit_prob = torch.sigmoid(vit_logits)
        self.last_band_prob = torch.sigmoid(band_logits)
        self.last_gate_weights = weights
        if self.fusion_mode == "residual":
            band_weight = weights[:, None, 1:2]
            correction = self.corr_logit_limit * band_weight * torch.tanh(band_logits - vit_logits)
            return vit_logits + correction
        if self.fusion_mode != "mixture":
            raise ValueError(f"Unknown fusion_mode={self.fusion_mode!r}")
        return weights[:, None, 0:1] * vit_logits + weights[:, None, 1:2] * band_logits

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def run(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = base_dir / raw_root
    out_dir = base_dir / "nmc_branchbands_valid_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    set_seed(int(args.seed))

    cfg = NMCBranchBandsConfig(
        base_dir=base_dir,
        raw_root=raw_root,
        output_prefix=str(args.output_prefix),
        seed=int(args.seed),
        train_profiles=_split_list(args.train_profiles),
        test_profiles=_split_list(args.test_profiles),
        window_len=int(args.window_len),
        stride=int(args.stride),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        kernel_size=int(args.kernel_size),
        dropout=float(args.dropout),
        lambda_rex=float(args.lambda_rex),
        rex_group=str(args.rex_group),
        loss_kind=str(args.loss_kind),
        huber_beta=float(args.huber_beta),
        lambda_smooth=float(args.lambda_smooth),
        endpoint_loss_weight=float(args.endpoint_loss_weight),
        lambda_worst=float(args.lambda_worst),
        num_workers=int(args.num_workers),
        print_every=int(args.print_every),
    )
    cfg.valid_profiles = _split_list(args.valid_profiles)

    files = find_csv_files(raw_root)
    start_audit = write_start_audit(files, out_dir / f"{args.output_prefix}_file_start_audit.csv")
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{args.output_prefix}_decomposition_params.csv", index=False)

    frames = build_feature_frames(cfg, files, r0_df)
    feature_cols = _selected_feature_columns(str(args.feature_set))
    available = set().union(*(set(f.columns) for split in frames.values() for f in split))
    missing = [c for c in feature_cols if c not in available]
    if missing:
        raise KeyError(f"Missing feature columns for {args.feature_set}: {missing}")
    schema = write_input_schema(feature_cols, cfg, out_dir / f"{args.output_prefix}_input_schema.csv")
    leakage = write_leakage_audit(feature_cols, raw_source_columns, cfg, out_dir / f"{args.output_prefix}_leakage_audit.csv")

    scaled, _ = make_scaled_frames_for_ablation(frames, feature_cols)
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(args.batch_size)
    base_cfg.dataloader_num_workers = int(args.num_workers)
    base_cfg.dataloader_prefetch_factor = int(args.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(args.num_workers) > 0

    train_ds = AugmentedSequenceWindowDataset(
        scaled["train"],
        feature_cols,
        int(args.window_len),
        int(args.stride),
        target_label="physical",
        window_feature_mode=str(args.window_feature_mode),
    )
    valid_ds = AugmentedWindowDataset(
        scaled["valid"],
        feature_cols,
        int(args.window_len),
        1,
        target_label="physical",
        window_feature_mode=str(args.window_feature_mode),
    )
    test_ds = AugmentedWindowDataset(
        scaled["test"],
        feature_cols,
        int(args.window_len),
        1,
        target_label="physical",
        window_feature_mode=str(args.window_feature_mode),
    )
    if min(len(train_ds), len(valid_ds), len(test_ds)) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")

    input_dim = augmented_input_dim(len(feature_cols), str(args.window_feature_mode))
    model_name = str(args.output_prefix)
    if str(args.model_kind) == "obs_fusion":
        model = ObservabilityFusionTCN(
            input_dim=input_dim,
            feature_cols=feature_cols,
            window_feature_mode=str(args.window_feature_mode),
            hidden_size=int(args.hidden_size),
            layers=int(args.layers),
            kernel_size=int(args.kernel_size),
            norm_kind=str(args.norm_kind),
            dropout=float(args.dropout),
            vit_hidden=int(args.vit_hidden),
            vit_layers=int(args.vit_layers),
            gate_prior_vit=float(args.gate_prior_vit),
            fusion_mode=str(args.fusion_mode),
            corr_logit_limit=float(args.corr_logit_limit),
        ).to(device)
    elif str(args.model_kind) == "tcn":
        model = DeepNoLeakTCN(
            input_dim=input_dim,
            hidden_size=int(args.hidden_size),
            layers=int(args.layers),
            kernel_size=int(args.kernel_size),
            norm_kind=str(args.norm_kind),
            dropout=float(args.dropout),
        ).to(device)
    elif str(args.model_kind) == "tcn_temp_affine":
        model = DeepNoLeakTempAffineTCN(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=int(args.hidden_size),
            layers=int(args.layers),
            kernel_size=int(args.kernel_size),
            norm_kind=str(args.norm_kind),
            dropout=float(args.dropout),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_kind={args.model_kind!r}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_loader = temperature_balanced_loader(train_ds, base_cfg, shuffle=True)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history_rows = []
    valid_rows = []
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        mean_losses = []
        rex_losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            pred = model.forward_sequence(x)
            loss, mean_loss, rex_loss, _smooth_loss, _by_group = temp_balanced_rex_loss(
                pred,
                y,
                meta,
                float(args.lambda_rex),
                str(args.rex_group),
                str(args.loss_kind),
                float(args.huber_beta),
                float(args.lambda_smooth),
                float(args.endpoint_loss_weight),
                float(args.lambda_worst),
            )
            if float(args.branch_aux_weight) > 0.0 and hasattr(model, "last_vit_prob") and hasattr(model, "last_band_prob"):
                vit_aux, _vit_mean, _vit_rex, _vit_smooth, _vit_groups = temp_balanced_rex_loss(
                    model.last_vit_prob,
                    y,
                    meta,
                    float(args.lambda_rex),
                    str(args.rex_group),
                    str(args.loss_kind),
                    float(args.huber_beta),
                    float(args.lambda_smooth),
                    float(args.endpoint_loss_weight),
                    float(args.lambda_worst),
                )
                band_aux, _band_mean, _band_rex, _band_smooth, _band_groups = temp_balanced_rex_loss(
                    model.last_band_prob,
                    y,
                    meta,
                    float(args.lambda_rex),
                    str(args.rex_group),
                    str(args.loss_kind),
                    float(args.huber_beta),
                    float(args.lambda_smooth),
                    float(args.endpoint_loss_weight),
                    float(args.lambda_worst),
                )
                loss = loss + float(args.branch_aux_weight) * (vit_aux + band_aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mean_losses.append(float(mean_loss.detach().cpu()))
            rex_losses.append(float(rex_loss.detach().cpu()))
        row = {
            "epoch": int(ep),
            "loss": float(np.mean(losses)),
            "mean_loss": float(np.mean(mean_losses)),
            "rex_var": float(np.mean(rex_losses)),
        }
        if ep == 1 or ep == int(args.epochs) or ep % max(1, int(args.eval_every)) == 0:
            valid_pred = predict(model, valid_loader)
            temp_table = _valid_temp_table(valid_pred)
            score = _selector_score(temp_table, str(args.selector))
            row["valid_selector_score"] = float(score)
            for _, r in temp_table.iterrows():
                valid_rows.append({"epoch": int(ep), **r.to_dict()})
                row[f"valid_mae_T{float(r['temperature_C']):g}"] = float(r["MAE_pct"])
            if score < best_score:
                best_score = float(score)
                best_epoch = int(ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"{model_name} epoch={ep} loss={row['loss']:.5f} valid_score={score:.3f}% best_ep={best_epoch}",
                flush=True,
            )
        history_rows.append(row)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    valid_pred = _attach_for_metrics(predict(model, valid_loader), frames, feature_cols, "valid", model_name)
    test_pred = _attach_for_metrics(predict(model, test_loader), frames, feature_cols, "test", model_name)
    for pred in (valid_pred, test_pred):
        pred["seed"] = int(args.seed)
        pred["selected_epoch"] = int(best_epoch)
        pred["selector"] = str(args.selector)
        pred["feature_set"] = str(args.feature_set)

    history = pd.DataFrame(history_rows)
    valid_trace = pd.DataFrame(valid_rows)
    test_overall = _overall_metrics(test_pred)
    test_by_temp = variance_by_temperature(test_pred)
    test_by_traj = metrics_by_trajectory(test_pred)
    test_focus = focus_metrics(test_pred, cfg, model_name)
    valid_by_temp = variance_by_temperature(valid_pred)

    history.to_csv(out_dir / f"{args.output_prefix}_history.csv", index=False)
    valid_trace.to_csv(out_dir / f"{args.output_prefix}_valid_trace.csv", index=False)
    valid_pred.to_csv(out_dir / f"{args.output_prefix}_valid_prediction_rows.csv.gz", index=False, compression="gzip")
    test_pred.to_csv(out_dir / f"{args.output_prefix}_test_prediction_rows.csv.gz", index=False, compression="gzip")
    test_overall.to_csv(out_dir / f"{args.output_prefix}_test_overall.csv", index=False)
    test_by_temp.to_csv(out_dir / f"{args.output_prefix}_test_by_temperature.csv", index=False)
    test_by_traj.to_csv(out_dir / f"{args.output_prefix}_test_by_trajectory.csv", index=False)
    test_focus.to_csv(out_dir / f"{args.output_prefix}_test_focus.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{args.output_prefix}_valid_by_temperature.csv", index=False)

    metadata = {
        **asdict(cfg),
        "valid_profiles": list(cfg.valid_profiles),
        "feature_set": str(args.feature_set),
        "model_kind": str(args.model_kind),
        "feature_columns": feature_cols,
        "input_feature_dim": int(input_dim),
        "window_feature_mode": str(args.window_feature_mode),
        "branch_aux_weight": float(args.branch_aux_weight),
        "gate_prior_vit": float(args.gate_prior_vit),
        "fusion_mode": str(args.fusion_mode),
        "corr_logit_limit": float(args.corr_logit_limit),
        "selected_epoch": int(best_epoch),
        "best_valid_selector_score_pct": float(best_score),
        "uses_soc_input": False,
        "uses_cumulative_input": False,
        "uses_explicit_current_integration": False,
        "current_usage": "instantaneous excitation and causal voltage-response features only",
        "train_windows": int(len(train_ds)),
        "valid_windows": int(len(valid_ds)),
        "test_windows": int(len(test_ds)),
    }
    (out_dir / f"{args.output_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    report = [
        "# NMC BranchBands Valid-Selector Screening",
        "",
        f"- Model kind: `{args.model_kind}`",
        f"- Feature set: `{args.feature_set}`",
        f"- Selected epoch: {best_epoch}",
        f"- Selector: `{args.selector}` on valid profiles `{args.valid_profiles}`",
        "- Test profile FUDS was not used for checkpoint selection.",
        "- Strict NoCC: no SOC input, no cumulative Ah/progress/time input, no explicit current-integration SOC update.",
        "",
        "## Test Overall",
        test_overall.to_string(index=False),
        "",
        "## Test By Temperature",
        test_by_temp.to_string(index=False),
        "",
        "## Leakage Audit",
        leakage.to_string(index=False),
        "",
        "## Input Schema",
        schema.to_string(index=False),
        "",
        "## Start Audit",
        start_audit.to_string(index=False),
    ]
    (out_dir / f"{args.output_prefix}_report.md").write_text("\n".join(report), encoding="utf-8")

    print("Selected epoch:", best_epoch, "valid_score_pct:", f"{best_score:.3f}", flush=True)
    print("Test overall:")
    print(test_overall.to_string(index=False), flush=True)
    print("Test by temperature:")
    print(test_by_temp.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC BranchBands TCN with valid-only checkpoint selection.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default="nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    p.add_argument("--output-prefix", default="nmc_branchbands_valid_selector")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-profiles", default="DST,US06")
    p.add_argument("--valid-profiles", default="VALIDATION")
    p.add_argument("--test-profiles", default="FUDS")
    p.add_argument("--feature-set", default="branch_bands")
    p.add_argument("--window-feature-mode", default="delta_start_time")
    p.add_argument("--model-kind", choices=["tcn", "tcn_temp_affine", "obs_fusion"], default="tcn")
    p.add_argument("--window-len", type=int, default=150)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--vit-hidden", type=int, default=64)
    p.add_argument("--vit-layers", type=int, default=5)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--norm-kind", default="channel")
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--rex-group", default="temperature_drive")
    p.add_argument("--loss-kind", default="huber")
    p.add_argument("--huber-beta", type=float, default=0.02)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    p.add_argument("--lambda-worst", type=float, default=0.0)
    p.add_argument("--branch-aux-weight", type=float, default=0.25)
    p.add_argument("--gate-prior-vit", type=float, default=0.8)
    p.add_argument("--fusion-mode", choices=["mixture", "residual"], default="mixture")
    p.add_argument("--corr-logit-limit", type=float, default=0.7)
    p.add_argument("--selector", choices=["valid_worst_temp", "valid_mean_temp", "valid_25"], default="valid_worst_temp")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
