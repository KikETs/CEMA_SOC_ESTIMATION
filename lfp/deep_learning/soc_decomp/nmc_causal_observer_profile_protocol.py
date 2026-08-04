from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .neural_ecm_observer import NeuralECMObserver
from .nmc_branchbands_experiment import (
    NMCBranchBandsConfig,
    build_feature_frames,
    estimate_r0_by_temperature,
    find_csv_files,
    parse_profile,
    write_start_audit,
)
from .runtime import configure_torch_runtime, device


TARGETS = {0.0: 1.0, 25.0: 0.7, 45.0: 0.3}


@dataclass
class NMCCausalObserverConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_samsung_inr_18650_2ah_raw/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = "nmc_causal_observer_e1_seed0"
    seed: int = 0
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("VALIDATION",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    train_mode: str = "chunk"
    prefix_len: int = 2048
    epochs: int = 60
    batch_size: int = 192
    chunk_len: int = 256
    chunk_stride: int = 128
    lr: float = 8e-4
    weight_decay: float = 1e-4
    lambda_voltage: float = 0.20
    lambda_param: float = 0.02
    lambda_rex: float = 0.50
    lambda_worst: float = 0.10
    correction_limit: float = 0.04
    init_soc_noise: float = 0.01
    state_noise: float = 0.005
    q_ref_ah: float = 2.0
    dt_sec: float = 1.0
    num_workers: int = 0
    print_every: int = 5
    eval_every: int = 10
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    if value.strip().upper() in {"", "NONE", "NULL"}:
        return ()
    return tuple(x.strip() for x in value.split(",") if x.strip())


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _add_observer_columns(frames: dict[str, list[pd.DataFrame]], cfg: NMCCausalObserverConfig) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    for split, split_frames in frames.items():
        for frame in split_frames:
            f = frame.copy()
            # NMC files use negative current for discharge. The observer state equation
            # uses positive discharge current, so convert sign explicitly.
            f["I_obs"] = -pd.to_numeric(f["I_raw"], errors="coerce").astype(np.float32)
            f["Q_ref_Ah"] = np.float32(cfg.q_ref_ah)
            f["SOC0_online"] = np.float32(f["SOC_physical"].iloc[0])
            out[split].append(f.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True))
    return out


class NMCObserverChunkDataset(Dataset):
    def __init__(self, frames: list[pd.DataFrame], chunk_len: int, stride: int):
        self.frames = []
        self.index: list[tuple[int, int, int]] = []
        self.chunk_len = int(chunk_len)
        self.stride = int(stride)
        for fi, frame in enumerate(frames):
            f = frame.reset_index(drop=True)
            self.frames.append(
                {
                    "I": np.ascontiguousarray(f["I_obs"].to_numpy(np.float32)),
                    "V": np.ascontiguousarray(f["V_raw"].to_numpy(np.float32)),
                    "T": np.ascontiguousarray(f["T"].to_numpy(np.float32)),
                    "soc": np.ascontiguousarray(f["SOC_physical"].to_numpy(np.float32)),
                    "temperature": float(f["temperature"].iloc[0]),
                    "drive_cycle": str(f["drive_cycle"].iloc[0]),
                    "trajectory_id": str(f["trajectory_id"].iloc[0]),
                }
            )
            n = len(f)
            if n < self.chunk_len:
                continue
            for start in range(0, n - self.chunk_len + 1, self.stride):
                end = start + self.chunk_len - 1
                y = self.frames[-1]["soc"][start : end + 1]
                if np.isfinite(y).all():
                    self.index.append((fi, start, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        meta = {
            "temperature": f["temperature"],
            "drive_cycle": f["drive_cycle"],
            "trajectory_id": f["trajectory_id"],
        }
        sl = slice(start, end + 1)
        return (
            torch.from_numpy(f["I"][sl]),
            torch.from_numpy(f["V"][sl]),
            torch.from_numpy(f["T"][sl]),
            torch.from_numpy(f["soc"][sl]),
            meta,
        )


def make_balanced_loader(ds: NMCObserverChunkDataset, cfg: NMCCausalObserverConfig) -> DataLoader:
    temps = np.asarray([ds.frames[fi]["temperature"] for fi, _, _ in ds.index], dtype=np.float32)
    profiles = np.asarray([ds.frames[fi]["drive_cycle"] for fi, _, _ in ds.index])
    keys = [f"{float(t):g}_{p}" for t, p in zip(temps, profiles)]
    unique, counts = np.unique(keys, return_counts=True)
    count_map = {str(k): int(c) for k, c in zip(unique, counts)}
    weights = np.asarray([1.0 / count_map[str(k)] for k in keys], dtype=np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
    )


def _meta_temps(meta) -> torch.Tensor:
    temps = meta["temperature"]
    if torch.is_tensor(temps):
        return temps.to(device=device, dtype=torch.float32)
    return torch.as_tensor(temps, device=device, dtype=torch.float32)


def observer_loss(model: NeuralECMObserver, batch, cfg: NMCCausalObserverConfig):
    I, V, T, y, meta = batch
    I = I.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    V = V.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    T = T.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    soc0 = y[:, 0]
    if model.training and cfg.init_soc_noise > 0:
        soc0 = (soc0 + torch.randn_like(soc0) * float(cfg.init_soc_noise)).clamp(0.0, 1.0)
    out = model.rollout(I, V, T, soc0=soc0, state_noise=float(cfg.state_noise))
    soc_pred = out["soc"]
    by_sample = torch.mean(torch.abs(soc_pred - y), dim=1)
    temps = _meta_temps(meta)
    temp_losses = []
    for temp in torch.unique(temps):
        temp_losses.append(by_sample[temps == temp].mean())
    stack = torch.stack(temp_losses) if temp_losses else by_sample.mean().view(1)
    l_soc = stack.mean()
    l_rex = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
    l_worst = stack.max()
    l_v = F.smooth_l1_loss(out["voltage"], V, beta=0.03)
    l_param = model.regularization(I.device)
    total = (
        l_soc
        + float(cfg.lambda_voltage) * l_v
        + float(cfg.lambda_param) * l_param
        + float(cfg.lambda_rex) * l_rex
        + float(cfg.lambda_worst) * l_worst
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_soc": float(l_soc.detach().cpu()),
        "loss_voltage": float(l_v.detach().cpu()),
        "loss_rex": float(l_rex.detach().cpu()),
        "loss_worst": float(l_worst.detach().cpu()),
        "loss_param": float(l_param.detach().cpu()),
    }


def full_trajectory_loss(
    model: NeuralECMObserver,
    frame: pd.DataFrame,
    cfg: NMCCausalObserverConfig,
    max_len: int | None = None,
):
    f = frame.reset_index(drop=True)
    if max_len is not None:
        f = f.iloc[: max(2, min(int(max_len), len(f)))].reset_index(drop=True)
    I = torch.as_tensor(f["I_obs"].to_numpy(np.float32)[None, :], device=device)
    V = torch.as_tensor(f["V_raw"].to_numpy(np.float32)[None, :], device=device)
    T = torch.as_tensor(f["T"].to_numpy(np.float32)[None, :], device=device)
    y = torch.as_tensor(f["SOC_physical"].to_numpy(np.float32)[None, :], device=device)
    soc0 = y[:, 0]
    if model.training and cfg.init_soc_noise > 0:
        soc0 = (soc0 + torch.randn_like(soc0) * float(cfg.init_soc_noise)).clamp(0.0, 1.0)
    out = model.rollout(I, V, T, soc0=soc0, state_noise=float(cfg.state_noise))
    l_soc = torch.mean(torch.abs(out["soc"] - y))
    l_endpoint = torch.mean(torch.abs(out["soc"][:, -1] - y[:, -1]))
    l_v = F.smooth_l1_loss(out["voltage"], V, beta=0.03)
    l_param = model.regularization(I.device)
    total = (
        l_soc
        + 0.25 * l_endpoint
        + float(cfg.lambda_voltage) * l_v
        + float(cfg.lambda_param) * l_param
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_soc": float(l_soc.detach().cpu()),
        "loss_endpoint": float(l_endpoint.detach().cpu()),
        "loss_voltage": float(l_v.detach().cpu()),
        "loss_rex": 0.0,
        "loss_worst": 0.0,
        "loss_param": float(l_param.detach().cpu()),
    }


@torch.no_grad()
def predict_trajectories(model: NeuralECMObserver, frames: list[pd.DataFrame], model_name: str) -> pd.DataFrame:
    rows = []
    model.eval()
    for frame in frames:
        f = frame.reset_index(drop=True)
        I = torch.as_tensor(f["I_obs"].to_numpy(np.float32)[None, :], device=device)
        V = torch.as_tensor(f["V_raw"].to_numpy(np.float32)[None, :], device=device)
        T = torch.as_tensor(f["T"].to_numpy(np.float32)[None, :], device=device)
        soc0 = torch.as_tensor([float(f["SOC0_online"].iloc[0])], device=device, dtype=torch.float32)
        out = model.rollout(I, V, T, soc0=soc0)
        pred = out["soc"].detach().cpu().numpy()[0]
        vpred = out["voltage"].detach().cpu().numpy()[0]
        df = pd.DataFrame(
            {
                "model_name": model_name,
                "trajectory_id": f["trajectory_id"].to_numpy(),
                "file_name": f["file_name"].to_numpy(),
                "temperature_C": f["temperature"].to_numpy(np.float32),
                "drive_cycle": f["drive_cycle"].to_numpy(),
                "end_index": f["end_index"].to_numpy(np.int64),
                "I_raw": f["I_raw"].to_numpy(np.float32),
                "I_obs_positive_discharge": f["I_obs"].to_numpy(np.float32),
                "V_raw": f["V_raw"].to_numpy(np.float32),
                "V_pred": vpred,
                "y_true": f["SOC_physical"].to_numpy(np.float32),
                "y_pred": pred,
            }
        )
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = np.abs(df["error"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metrics_by_temperature(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for (model, temp), g in pred.groupby(["model_name", "temperature_C"]):
        err = g["error"].to_numpy(np.float64)
        rows.append(
            {
                "model_name": model,
                "temperature_C": float(temp),
                "n_points": int(len(g)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "temperature_C"]).reset_index(drop=True)


def overall_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for model, g in pred.groupby("model_name"):
        err = g["error"].to_numpy(np.float64)
        rows.append(
            {
                "model_name": model,
                "n_points": int(len(g)),
                "MAE_pct": float(np.mean(np.abs(err)) * 100.0),
                "RMSE_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
                "bias_pct": float(np.mean(err) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def focus_table(by_temp: pd.DataFrame) -> pd.DataFrame:
    if by_temp.empty:
        return pd.DataFrame()
    rows = []
    for model, g in by_temp.groupby("model_name"):
        target_met = True
        target_worst = 0.0
        details = []
        for temp, target in TARGETS.items():
            row = g[np.isclose(g["temperature_C"].astype(float), temp)]
            if row.empty:
                target_met = False
                details.append(f"{temp:g}C missing")
                continue
            mae = float(row["MAE_pct"].iloc[0])
            target_worst = max(target_worst, mae / target)
            if mae >= target:
                target_met = False
                details.append(f"{temp:g}C {mae:.3f}>={target:.3f}")
        rows.append(
            {
                "model_name": model,
                "target_met": bool(target_met),
                "target_norm_worst": float(target_worst),
                "failure_detail": "; ".join(details) if details else "pass",
            }
        )
    return pd.DataFrame(rows)


def write_audit(cfg: NMCCausalObserverConfig, out_path: Path) -> pd.DataFrame:
    rows = [
        {
            "audit_item": "track_type",
            "status": "DECLARED",
            "detail": "This is a causal observer main-track candidate, not a strict NoCC model.",
        },
        {
            "audit_item": "explicit_current_integration_state_update",
            "status": "DECLARED_TRUE",
            "detail": "SOC is propagated by a learned charge-conservation state update using positive-discharge current.",
        },
        {
            "audit_item": "current_sign_convention",
            "status": "PASS",
            "detail": "NMC raw discharge current is negative; observer input uses I_obs=-I_raw so positive current means discharge.",
        },
        {
            "audit_item": "initial_soc",
            "status": "DECLARED_TRUE",
            "detail": "Online trajectory inference starts from the file initial SOC, 0.8 for this NMC dataset. This must be reported as an initial-condition assumption.",
        },
        {
            "audit_item": "forbidden_nocc_claims",
            "status": "PASS",
            "detail": "Do not call this NoCC and do not claim current is unused.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_report(
    cfg: NMCCausalObserverConfig,
    out_dir: Path,
    history: pd.DataFrame,
    valid_by_temp: pd.DataFrame,
    test_by_temp: pd.DataFrame,
    test_focus: pd.DataFrame,
) -> None:
    lines = [
        "# NMC Causal Observer Profile Protocol",
        "",
        "## Purpose",
        "",
        "This run is intentionally not a strict NoCC experiment. It tests a defensible main-track alternative: a causal SOC observer that openly uses current-driven charge conservation plus voltage-residual correction.",
        "",
        "## Setup",
        "",
        f"- Train profiles: {', '.join(cfg.train_profiles)}",
        f"- Valid profiles: {', '.join(cfg.valid_profiles) if cfg.valid_profiles else '(none)'}",
        f"- Test profiles: {', '.join(cfg.test_profiles)}",
        "- Temperatures: 0C, 25C, 45C",
        f"- Epochs: {cfg.epochs}",
        f"- Training mode: {cfg.train_mode}",
        f"- Prefix length if prefix mode: {cfg.prefix_len}",
        f"- Chunk length/stride: {cfg.chunk_len}/{cfg.chunk_stride}",
        f"- Nominal Q_ref: {cfg.q_ref_ah} Ah",
        f"- Initial SOC assumption: file-level initial SOC, 0.8 for NMC",
        "- Current convention: `I_obs = -I_raw`, so positive observer current means discharge",
        "",
        "## Adoption Rule",
        "",
        "For this first implementation the paper-safe rule is fixed final epoch, not train-25C checkpoint selection and not Stage2 correction. Validation is reported for diagnosis only unless a selector is explicitly added later.",
        "",
        "If `train_mode=full`, each training trajectory starts only from the file-level initial SOC and is rolled forward to the end. If `train_mode=prefix`, each update also starts only from the file-level initial SOC but uses a bounded prefix length to control compute. If `train_mode=chunk`, chunk-start SOC is used only as a truncated-BPTT training convenience and should be treated as a smoke/diagnostic mode rather than the final online protocol.",
        "",
        "## Validation By Temperature",
        "",
        valid_by_temp.to_markdown(index=False, floatfmt=".3f") if not valid_by_temp.empty else "(no validation split)",
        "",
        "## FUDS Test By Temperature",
        "",
        test_by_temp.to_markdown(index=False, floatfmt=".3f") if not test_by_temp.empty else "(no test rows)",
        "",
        "## Target Verdict",
        "",
        test_focus.to_markdown(index=False, floatfmt=".3f") if not test_focus.empty else "(no focus rows)",
        "",
        "## Claim Guardrails",
        "",
        "- This model uses current integration; it is not a NoCC result.",
        "- Do not compare it as proof that the NoCC constraint is unnecessary.",
        "- The valid comparison is: causal observer main-track vs strict NoCC ablation.",
        "- If this passes, the claim is an online observer with known/estimated initial SOC, not a stateless voltage-only SOC estimator.",
        "",
        "## Last Training Rows",
        "",
        history.tail(10).to_markdown(index=False, floatfmt=".5f") if not history.empty else "(no history)",
        "",
    ]
    (out_dir / f"{cfg.output_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: NMCCausalObserverConfig) -> dict[str, pd.DataFrame]:
    configure_torch_runtime()
    set_seed(cfg.seed)
    out_dir = cfg.base_dir / "nmc_causal_observer_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_csv_files(cfg.raw_root)
    start_audit = write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    train_files = []
    for path in files:
        head = pd.read_csv(path, nrows=2)
        if parse_profile(path, head) in cfg.train_profiles:
            train_files.append(path)
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    branch_cfg = NMCBranchBandsConfig(
        base_dir=cfg.base_dir,
        raw_root=cfg.raw_root,
        output_prefix=cfg.output_prefix,
        seed=cfg.seed,
        train_profiles=cfg.train_profiles,
        test_profiles=cfg.test_profiles,
        window_len=150,
        stride=3,
        v_corr_tau_s=cfg.v_corr_tau_s,
        v_pol_mid_tau_s=cfg.v_pol_mid_tau_s,
        v_pol_slow_tau_s=cfg.v_pol_slow_tau_s,
        v_hys_tau_s=cfg.v_hys_tau_s,
    )
    setattr(branch_cfg, "valid_profiles", cfg.valid_profiles)
    frames = _add_observer_columns(build_feature_frames(branch_cfg, files, r0_df), cfg)

    train_ds = NMCObserverChunkDataset(frames["train"], chunk_len=cfg.chunk_len, stride=cfg.chunk_stride)
    train_loader = None
    if cfg.train_mode == "chunk":
        if len(train_ds) == 0:
            raise RuntimeError("No train chunks for NMC causal observer.")
        train_loader = make_balanced_loader(train_ds, cfg)
    elif cfg.train_mode in {"full", "prefix"}:
        if not frames["train"]:
            raise RuntimeError("No train trajectories for NMC causal observer.")
    else:
        raise ValueError(f"Unknown train_mode={cfg.train_mode!r}; use 'chunk', 'prefix', or 'full'.")

    model = NeuralECMObserver(
        q_ref_ah=cfg.q_ref_ah,
        dt_sec=cfg.dt_sec,
        correction_limit=cfg.correction_limit,
        use_current_integration=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history_rows = []
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        meters = []
        if cfg.train_mode == "chunk":
            assert train_loader is not None
            for batch in train_loader:
                loss, parts = observer_loss(model, batch, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                meters.append(parts)
        elif cfg.train_mode == "full":
            order = list(range(len(frames["train"])))
            random.shuffle(order)
            for idx in order:
                loss, parts = full_trajectory_loss(model, frames["train"][idx], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                meters.append(parts)
        else:
            order = list(range(len(frames["train"])))
            random.shuffle(order)
            for idx in order:
                loss, parts = full_trajectory_loss(model, frames["train"][idx], cfg, max_len=cfg.prefix_len)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                meters.append(parts)
        row = {
            "epoch": epoch,
            **{k: float(np.mean([m[k] for m in meters])) for k in meters[0]},
        }
        if epoch == 1 or epoch == cfg.epochs or epoch % max(1, cfg.print_every) == 0:
            print(
                f"{cfg.output_prefix} epoch={epoch} "
                f"loss={row['loss']:.5f} soc={row['loss_soc']:.5f} voltage={row['loss_voltage']:.5f}"
            )
        history_rows.append(row)

    model_name = "CausalNeuralECMObserver_fixed_final"
    valid_pred = predict_trajectories(model, frames["valid"], model_name)
    test_pred = predict_trajectories(model, frames["test"], model_name)
    history = pd.DataFrame(history_rows)
    valid_by_temp = metrics_by_temperature(valid_pred)
    test_by_temp = metrics_by_temperature(test_pred)
    valid_overall = overall_metrics(valid_pred)
    test_overall = overall_metrics(test_pred)
    test_focus = focus_table(test_by_temp)
    audit = write_audit(cfg, out_dir / f"{cfg.output_prefix}_observer_audit.csv")

    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_r0_by_temperature.csv", index=False)
    valid_pred.to_csv(out_dir / f"{cfg.output_prefix}_valid_predictions.csv", index=False)
    test_pred.to_csv(out_dir / f"{cfg.output_prefix}_test_predictions.csv", index=False)
    valid_by_temp.to_csv(out_dir / f"{cfg.output_prefix}_valid_by_temperature.csv", index=False)
    test_by_temp.to_csv(out_dir / f"{cfg.output_prefix}_test_by_temperature.csv", index=False)
    valid_overall.to_csv(out_dir / f"{cfg.output_prefix}_valid_overall.csv", index=False)
    test_overall.to_csv(out_dir / f"{cfg.output_prefix}_test_overall.csv", index=False)
    test_focus.to_csv(out_dir / f"{cfg.output_prefix}_focus.csv", index=False)
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(
        json.dumps({**asdict(cfg), "device": str(device), "n_train_chunks": len(train_ds)}, default=str, indent=2),
        encoding="utf-8",
    )
    write_report(cfg, out_dir, history, valid_by_temp, test_by_temp, test_focus)
    print(test_by_temp.to_string(index=False))
    print(test_focus.to_string(index=False))
    return {
        "start_audit": start_audit,
        "audit": audit,
        "history": history,
        "valid_by_temperature": valid_by_temp,
        "test_by_temperature": test_by_temp,
        "test_focus": test_focus,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMC profile-split causal SOC observer protocol.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=NMCCausalObserverConfig.raw_root)
    p.add_argument("--output-prefix", default=NMCCausalObserverConfig.output_prefix)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-profiles", default="DST,US06")
    p.add_argument("--valid-profiles", default="VALIDATION")
    p.add_argument("--test-profiles", default="FUDS")
    p.add_argument("--train-mode", choices=("chunk", "prefix", "full"), default="chunk")
    p.add_argument("--prefix-len", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--chunk-stride", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lambda-voltage", type=float, default=0.20)
    p.add_argument("--lambda-rex", type=float, default=0.50)
    p.add_argument("--lambda-worst", type=float, default=0.10)
    p.add_argument("--correction-limit", type=float, default=0.04)
    p.add_argument("--q-ref-ah", type=float, default=2.0)
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NMCCausalObserverConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=args.output_prefix,
        seed=args.seed,
        train_profiles=_parse_csv_tuple(args.train_profiles),
        valid_profiles=_parse_csv_tuple(args.valid_profiles),
        test_profiles=_parse_csv_tuple(args.test_profiles),
        train_mode=args.train_mode,
        prefix_len=args.prefix_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
        chunk_stride=args.chunk_stride,
        lr=args.lr,
        lambda_voltage=args.lambda_voltage,
        lambda_rex=args.lambda_rex,
        lambda_worst=args.lambda_worst,
        correction_limit=args.correction_limit,
        q_ref_ah=args.q_ref_ah,
        print_every=args.print_every,
    )
    run(cfg)


if __name__ == "__main__":
    main()
