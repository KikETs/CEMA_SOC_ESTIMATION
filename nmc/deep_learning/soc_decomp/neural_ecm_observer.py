from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .models import collate_meta_to_frame
from .training import attach_prediction_features, build_prediction_feature_lookup
from .variance_control import variance_by_temperature, _overall_metrics
from .extrapolation_robustness import (
    EXP_SPECS,
    clone_cfg,
    load_filtered_feature_frames,
    exp_cfg,
    temp_key_to_c,
)


OUTSIDE_SPECS = {
    "Omit N10": {
        "train_temps": ("0", "10", "25", "50"),
        "omitted_temp_C": -10.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
    },
    "Omit 50": {
        "train_temps": ("N10", "0", "10", "25"),
        "omitted_temp_C": 50.0,
        "feature_dir": "decomposed_features_train_temp_minus10_0_10_25_50",
    },
}


@dataclass
class ECMSpec:
    name: str
    use_rex: bool = False
    lambda_v: float = 0.20
    lambda_rex: float = 0.5
    lambda_worst: float = 0.10
    lambda_param: float = 0.02
    correction_limit: float = 0.05
    init_soc_noise: float = 0.02
    state_noise: float = 0.01
    use_current_integration: bool = True


def ecm_exp_cfg(cfg: CFG | None, experiment: str) -> CFG:
    def copy_dynamic_ecm_attrs(dst):
        if cfg is not None:
            for k, v in vars(cfg).items():
                if k.startswith("ecm_"):
                    setattr(dst, k, v)
        return dst

    if experiment in EXP_SPECS:
        return copy_dynamic_ecm_attrs(exp_cfg(cfg, experiment))
    src = clone_cfg(cfg)
    spec = OUTSIDE_SPECS[experiment]
    src.smoke_mode = False
    src.use_existing_soc_cc_if_available = False
    src.use_existing_usable_if_available = False
    src.train_temps = spec["train_temps"]
    src.eval_temps = ("N10", "0", "10", "20", "25", "30", "40", "50")
    src.train_drives = ("DST", "US06")
    src.eval_drive = "FUDS"
    src.decomposed_dir = src.output_dir / spec["feature_dir"]
    return copy_dynamic_ecm_attrs(src)


def omitted_temp_for(experiment: str) -> float:
    if experiment in EXP_SPECS:
        return float(EXP_SPECS[experiment]["omitted_temp_C"])
    return float(OUTSIDE_SPECS[experiment]["omitted_temp_C"])


class ECMChunkDataset(Dataset):
    def __init__(self, frames, chunk_len=256, stride=128):
        self.frames = []
        self.index = []
        self.chunk_len = int(chunk_len)
        self.stride = int(stride)
        cols = [
            "I_raw",
            "V_raw",
            "temperature",
            "SOC_physical",
            "end_index",
            "trajectory_id",
            "drive_cycle",
            "Q_ref_Ah",
        ]
        for fi, frame in enumerate(frames):
            f = frame.reset_index(drop=True)
            if not set(cols).issubset(f.columns):
                missing = sorted(set(cols) - set(f.columns))
                raise KeyError(f"Missing columns for ECM observer dataset: {missing}")
            cache = {
                "I": np.ascontiguousarray(f["I_raw"].to_numpy(np.float32)),
                "V": np.ascontiguousarray(f["V_raw"].to_numpy(np.float32)),
                "T": np.ascontiguousarray(f["temperature"].to_numpy(np.float32)),
                "soc": np.ascontiguousarray(f["SOC_physical"].to_numpy(np.float32)),
                "end_index": f["end_index"].to_numpy(np.int64),
                "trajectory_id": f["trajectory_id"].to_numpy(),
                "drive_cycle": f["drive_cycle"].to_numpy(),
                "temperature": float(f["temperature"].iloc[0]),
                "q_ref": float(np.nanmedian(f["Q_ref_Ah"].to_numpy(np.float64))),
            }
            self.frames.append(cache)
            n = len(f)
            if n < self.chunk_len:
                continue
            for start in range(0, n - self.chunk_len + 1, self.stride):
                end = start + self.chunk_len - 1
                y = cache["soc"][start:end + 1]
                if np.isfinite(y).all():
                    self.index.append((fi, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        meta = {
            "trajectory_id": f["trajectory_id"][end],
            "drive_cycle": f["drive_cycle"][end],
            "temperature": f["temperature"],
            "end_index": int(f["end_index"][end]),
        }
        return (
            torch.from_numpy(f["I"][start:end + 1]),
            torch.from_numpy(f["V"][start:end + 1]),
            torch.from_numpy(f["T"][start:end + 1]),
            torch.from_numpy(f["soc"][start:end + 1]),
            meta,
        )


def dataset_temps(ds: ECMChunkDataset):
    return np.asarray([ds.frames[fi]["temperature"] for fi, _, _ in ds.index], dtype=np.float32)


def balanced_chunk_loader(ds: ECMChunkDataset, cfg: CFG, shuffle=True):
    if not shuffle:
        return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    temps = dataset_temps(ds)
    unique, counts = np.unique(temps, return_counts=True)
    count_map = {float(t): int(c) for t, c in zip(unique, counts)}
    weights = np.asarray([1.0 / count_map[float(t)] for t in temps], dtype=np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(ds, batch_size=int(cfg.batch_size), sampler=sampler, num_workers=0)


class MonotonicOCV(nn.Module):
    def __init__(self, n_knots=64):
        super().__init__()
        self.n_knots = int(n_knots)
        self.base = nn.Parameter(torch.tensor(2.45))
        init_inc = torch.full((self.n_knots - 1,), np.log(np.exp(1.10 / (self.n_knots - 1)) - 1.0))
        self.raw_inc = nn.Parameter(init_inc)
        self.t_corr = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 1))
        nn.init.zeros_(self.t_corr[-1].weight)
        nn.init.zeros_(self.t_corr[-1].bias)

    def forward(self, soc, temp_c):
        soc = soc.clamp(0.0, 1.0)
        knots = self.base + torch.cat(
            [soc.new_zeros(1), torch.cumsum(F.softplus(self.raw_inc), dim=0)], dim=0
        )
        pos = soc * (self.n_knots - 1)
        idx0 = pos.floor().long().clamp(0, self.n_knots - 2)
        frac = (pos - idx0.float()).unsqueeze(-1)
        v0 = knots[idx0].unsqueeze(-1)
        v1 = knots[(idx0 + 1).clamp(max=self.n_knots - 1)].unsqueeze(-1)
        ocv = v0 + frac * (v1 - v0)
        t_scaled = ((temp_c - 20.0) / 40.0).unsqueeze(-1)
        s = soc.unsqueeze(-1)
        return ocv.squeeze(-1) + 0.04 * torch.tanh(self.t_corr(torch.cat([s, t_scaled], dim=-1))).squeeze(-1)


class ECMParameterNet(nn.Module):
    def __init__(self, q_ref_ah=1.10):
        super().__init__()
        self.q_ref_ah = float(q_ref_ah)
        self.net = nn.Sequential(
            nn.Linear(3, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 15),
        )

    def _features(self, temp_c, soc, abs_i):
        return torch.stack([
            (temp_c - 20.0) / 40.0,
            soc.clamp(0.0, 1.0) * 2.0 - 1.0,
            (abs_i / 4.0).clamp(0.0, 2.0),
        ], dim=-1)

    def forward(self, temp_c, soc, abs_i, scales=None):
        raw = self.net(self._features(temp_c, soc, abs_i))
        q = self.q_ref_ah * (0.45 + 1.20 * torch.sigmoid(raw[..., 0]))
        r0 = 0.003 + 0.220 * torch.sigmoid(raw[..., 1])
        r1 = 0.001 + 0.180 * torch.sigmoid(raw[..., 2])
        tau1 = 1.0 + 249.0 * torch.sigmoid(raw[..., 3])
        r2 = 0.001 + 0.180 * torch.sigmoid(raw[..., 4])
        tau2 = 25.0 + 1975.0 * torch.sigmoid(raw[..., 5])
        hys_gain = 0.005 + 0.095 * torch.sigmoid(raw[..., 6])
        hys_tau = 50.0 + 2950.0 * torch.sigmoid(raw[..., 7])
        eta = 0.94 + 0.12 * torch.sigmoid(raw[..., 8])
        # Voltage-residual feedback should be a gentle observer correction, not
        # a direct voltage-to-SOC regressor. Large per-sample SOC gains accumulate
        # into severe drift over long FUDS trajectories.
        k_soc = 0.006 * torch.sigmoid(raw[..., 9])
        k_v1 = 0.035 * torch.sigmoid(raw[..., 10])
        k_v2 = 0.025 * torch.sigmoid(raw[..., 11])
        k_hys = 0.020 * torch.sigmoid(raw[..., 12])
        if scales is not None:
            q = q * torch.exp(scales[..., 0])
            r0 = r0 * torch.exp(scales[..., 1])
            r1 = r1 * torch.exp(scales[..., 2])
            r2 = r2 * torch.exp(scales[..., 2])
            tau1 = tau1 * torch.exp(scales[..., 3])
            tau2 = tau2 * torch.exp(scales[..., 3])
            hys_gain = hys_gain * torch.exp(scales[..., 4])
        return {
            "Q_eff": q,
            "R0": r0,
            "R1": r1,
            "tau1": tau1,
            "R2": r2,
            "tau2": tau2,
            "hys_gain": hys_gain,
            "hys_tau": hys_tau,
            "eta": eta,
            "k_soc": k_soc,
            "k_v1": k_v1,
            "k_v2": k_v2,
            "k_hys": k_hys,
        }


class NeuralECMObserver(nn.Module):
    def __init__(self, q_ref_ah=1.10, dt_sec=1.0, correction_limit=0.05, use_current_integration=True):
        super().__init__()
        self.dt_sec = float(dt_sec)
        self.correction_limit = float(correction_limit)
        self.use_current_integration = bool(use_current_integration)
        self.params = ECMParameterNet(q_ref_ah=q_ref_ah)
        self.ocv = MonotonicOCV()

    def _scale_params(self, scales, batch_size, device_):
        if scales is None:
            return None
        s = scales.to(device=device_, dtype=torch.float32)
        if s.ndim == 1:
            s = s.view(1, 5).expand(batch_size, 5)
        return s

    def rollout(self, I, V, T, soc0=None, train_perturb=0.0, state_noise=0.0, scales=None):
        B, L = I.shape
        scales = self._scale_params(scales, B, I.device)
        if soc0 is None:
            soc = I.new_ones(B)
        else:
            soc = soc0.to(device=I.device, dtype=torch.float32).clamp(0.0, 1.0)
        if self.training and train_perturb > 0:
            soc = (soc + torch.randn_like(soc) * float(train_perturb)).clamp(0.0, 1.0)
        v1 = I.new_zeros(B)
        v2 = I.new_zeros(B)
        hys = I.new_zeros(B)
        if self.training and state_noise > 0:
            v1 = v1 + torch.randn_like(v1) * float(state_noise)
            v2 = v2 + torch.randn_like(v2) * float(state_noise)
            hys = hys + torch.randn_like(hys) * float(state_noise) * 0.5
        soc_out, v_out, resid_out, hys_out = [], [], [], []
        param_trace = {k: [] for k in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]}
        for t in range(L):
            it = I[:, t]
            vt = V[:, t]
            tt = T[:, t]
            p = self.params(tt, soc, it.abs(), scales=scales)
            v_prior = self.ocv(soc, tt) + hys - v1 - v2 - it * p["R0"]
            resid = vt - v_prior
            dsoc = (p["k_soc"] * resid).clamp(-self.correction_limit, self.correction_limit)
            soc_c = (soc + dsoc).clamp(0.0, 1.0)
            v1_c = v1 - p["k_v1"] * resid
            v2_c = v2 - p["k_v2"] * resid
            hys_c = (hys + p["k_hys"] * resid).clamp(-0.20, 0.20)
            v_pred = self.ocv(soc_c, tt) + hys_c - v1_c - v2_c - it * p["R0"]
            soc_out.append(soc_c.unsqueeze(1))
            v_out.append(v_pred.unsqueeze(1))
            resid_out.append((vt - v_pred).unsqueeze(1))
            hys_out.append(hys_c.unsqueeze(1))
            for k in param_trace:
                param_trace[k].append(p[k].unsqueeze(1))
            if self.use_current_integration:
                soc = (soc_c - p["eta"] * it * (self.dt_sec / 3600.0) / p["Q_eff"].clamp_min(1e-4)).clamp(0.0, 1.0)
            else:
                soc = soc_c
            a1 = torch.exp(-self.dt_sec / p["tau1"].clamp_min(1e-3)).clamp(0.0, 0.99999)
            a2 = torch.exp(-self.dt_sec / p["tau2"].clamp_min(1e-3)).clamp(0.0, 0.99999)
            v1 = a1 * v1_c + (1.0 - a1) * p["R1"] * it
            v2 = a2 * v2_c + (1.0 - a2) * p["R2"] * it
            h_target = p["hys_gain"] * torch.tanh(3.0 * it)
            ah = torch.exp(-self.dt_sec / p["hys_tau"].clamp_min(1e-3)).clamp(0.0, 0.99999)
            hys = (ah * hys_c + (1.0 - ah) * h_target).clamp(-0.20, 0.20)
        out = {
            "soc": torch.cat(soc_out, dim=1),
            "voltage": torch.cat(v_out, dim=1),
            "voltage_residual": torch.cat(resid_out, dim=1),
            "hys": torch.cat(hys_out, dim=1),
            "params": {k: torch.cat(v, dim=1) for k, v in param_trace.items()},
        }
        return out

    def regularization(self, device_):
        temps = torch.linspace(-10.0, 50.0, 13, device=device_)
        socs = torch.linspace(0.05, 0.95, 19, device=device_)
        tt, ss = torch.meshgrid(temps, socs, indexing="ij")
        ii = torch.full_like(tt, 0.8)
        p = self.params(tt.reshape(-1), ss.reshape(-1), ii.reshape(-1))
        reg = tt.new_tensor(0.0)
        for key in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]:
            val = p[key].reshape(len(temps), len(socs))
            reg = reg + torch.mean((val[1:, :] - val[:-1, :]) ** 2)
            reg = reg + torch.mean((val[:, 1:] - val[:, :-1]) ** 2)
        r0 = p["R0"].reshape(len(temps), len(socs))
        reg = reg + 5.0 * torch.mean(F.relu(r0[1:, :] - r0[:-1, :]) ** 2)
        inc = F.softplus(self.ocv.raw_inc)
        reg = reg + 0.02 * torch.mean((inc[1:] - inc[:-1]) ** 2)
        return reg

    def parameter_curves(self):
        rows = []
        with torch.no_grad():
            temps = torch.tensor([-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0], device=next(self.parameters()).device)
            socs = torch.linspace(0.05, 0.95, 19, device=temps.device)
            for temp in temps:
                for soc in socs:
                    p = self.params(temp.view(1), soc.view(1), torch.tensor([0.8], device=temps.device))
                    row = {"temperature_C": float(temp.cpu()), "SOC": float(soc.cpu())}
                    for key in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]:
                        row[key] = float(p[key].detach().cpu().item())
                    row["OCV"] = float(self.ocv(soc.view(1), temp.view(1)).detach().cpu().item())
                    rows.append(row)
        return pd.DataFrame(rows)


def move_batch(I, V, T, y):
    return (
        I.to(device=device, dtype=torch.float32, non_blocking=True),
        V.to(device=device, dtype=torch.float32, non_blocking=True),
        T.to(device=device, dtype=torch.float32, non_blocking=True),
        y.to(device=device, dtype=torch.float32, non_blocking=True),
    )


def q_ref_from_frames(frames):
    vals = []
    for f in frames:
        if "Q_ref_Ah" in f.columns:
            vals.append(float(np.nanmedian(f["Q_ref_Ah"].to_numpy(np.float64))))
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else 1.10


def ecm_batch_loss(model, I, V, T, y, meta, spec: ECMSpec):
    out = model.rollout(
        I,
        V,
        T,
        soc0=y[:, 0],
        train_perturb=spec.init_soc_noise,
        state_noise=spec.state_noise,
    )
    soc_pred = out["soc"]
    l_by_sample = torch.mean(torch.abs(soc_pred - y), dim=1)
    temps = meta["temperature"]
    if torch.is_tensor(temps):
        temps_t = temps.to(device=soc_pred.device, dtype=torch.float32)
    else:
        temps_t = torch.as_tensor(temps, device=soc_pred.device, dtype=torch.float32)
    temp_losses = []
    loss_by_temp = {}
    for temp in torch.unique(temps_t):
        mask = temps_t == temp
        lt = l_by_sample[mask].mean()
        temp_losses.append(lt)
        loss_by_temp[float(temp.detach().cpu())] = lt.detach()
    if temp_losses:
        stack = torch.stack(temp_losses)
        l_soc = stack.mean()
        l_rex = stack.var(unbiased=False) if len(temp_losses) > 1 else stack.new_tensor(0.0)
        l_worst = stack.max()
    else:
        l_soc = l_by_sample.mean()
        l_rex = l_soc.new_tensor(0.0)
        l_worst = l_soc
    l_v = F.smooth_l1_loss(out["voltage"], V, beta=0.03)
    l_param = model.regularization(I.device)
    total = l_soc + spec.lambda_v * l_v + spec.lambda_param * l_param
    if spec.use_rex:
        total = total + spec.lambda_rex * l_rex + spec.lambda_worst * l_worst
    return total, {
        "loss_soc": float(l_soc.detach().cpu()),
        "loss_voltage": float(l_v.detach().cpu()),
        "loss_rex": float(l_rex.detach().cpu()),
        "loss_worst": float(l_worst.detach().cpu()),
        "loss_param": float(l_param.detach().cpu()),
    }, loss_by_temp


def train_ecm_model(feature_frames, cfg: CFG, spec: ECMSpec, experiment: str):
    chunk_len = int(getattr(cfg, "ecm_chunk_len", getattr(cfg, "window_len", 50)))
    chunk_stride = int(getattr(cfg, "ecm_chunk_stride", getattr(cfg, "stride", 1)))
    train_ds = ECMChunkDataset(feature_frames["train"], chunk_len=chunk_len, stride=chunk_stride)
    if len(train_ds) == 0:
        raise ValueError("No train chunks for NeuralECMObserver")
    train_loader = balanced_chunk_loader(train_ds, cfg, shuffle=True)
    model = NeuralECMObserver(
        q_ref_ah=q_ref_from_frames(feature_frames["train"]),
        dt_sec=float(getattr(cfg, "dt_sec", 1.0)),
        correction_limit=spec.correction_limit,
        use_current_integration=spec.use_current_integration,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(cfg, "ecm_lr", 8e-4)), weight_decay=1e-4)
    history = []
    temp_loss_rows = []
    epochs = int(getattr(cfg, "ecm_epochs", getattr(cfg, "lstm_epochs", 10)))
    print_every = max(1, int(getattr(cfg, "lstm_print_every", 5)))
    early_stop = bool(getattr(cfg, "ecm_early_stop", True))
    warmup_epochs = int(getattr(cfg, "ecm_plateau_warmup_epochs", 50))
    patience = int(getattr(cfg, "ecm_plateau_patience", 35))
    min_delta = float(getattr(cfg, "ecm_plateau_min_delta", 1e-4))
    monitor = str(getattr(cfg, "ecm_plateau_monitor", "loss_soc"))
    best_metric = float("inf")
    bad_epochs = 0
    for ep in range(1, epochs + 1):
        model.train()
        meters = []
        temp_meters = {}
        for I, V, T, y, meta in train_loader:
            I, V, T, y = move_batch(I, V, T, y)
            loss, parts, by_temp = ecm_batch_loss(model, I, V, T, y, meta, spec)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(getattr(cfg, "grad_clip", 1.0)))
            opt.step()
            meters.append({"total": float(loss.detach().cpu()), **parts})
            for temp, lt in by_temp.items():
                temp_meters.setdefault(temp, []).append(float(lt.cpu()))
        row = {
            "experiment": experiment,
            "model_name": spec.name,
            "epoch": ep,
            **{k: float(np.mean([m[k] for m in meters])) for k in meters[0]},
        }
        history.append(row)
        for temp, vals in temp_meters.items():
            temp_loss_rows.append({
                "experiment": experiment,
                "model_name": spec.name,
                "epoch": ep,
                "temperature_C": float(temp),
                "train_soc_loss": float(np.mean(vals)),
            })
        if ep == 1 or ep == epochs or ep % print_every == 0:
            print(
                f"{experiment} {spec.name} epoch={ep} "
                f"loss={row['total']:.5f} soc={row['loss_soc']:.5f} v={row['loss_voltage']:.5f}"
            )
        metric = float(row.get(monitor, row["loss_soc"]))
        if metric < best_metric - min_delta:
            best_metric = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
        if early_stop and ep >= warmup_epochs and bad_epochs >= patience:
            row["stopped_early"] = True
            row["stop_reason"] = (
                f"plateau monitor={monitor} best={best_metric:.6f} "
                f"patience={patience} min_delta={min_delta}"
            )
            history[-1] = row
            print(
                f"{experiment} {spec.name} early-stop at epoch={ep}: "
                f"{row['stop_reason']}"
            )
            break
    return model, pd.DataFrame(history), pd.DataFrame(temp_loss_rows)


@torch.no_grad()
def predict_full_trajectories(model, frames, model_name, scales_by_tid=None):
    rows = []
    model.eval()
    scales_by_tid = scales_by_tid or {}
    for frame in frames:
        f = frame.reset_index(drop=True)
        I = torch.as_tensor(f["I_raw"].to_numpy(np.float32)[None, :], device=device)
        V = torch.as_tensor(f["V_raw"].to_numpy(np.float32)[None, :], device=device)
        T = torch.as_tensor(f["temperature"].to_numpy(np.float32)[None, :], device=device)
        tid = f["trajectory_id"].iloc[0]
        scales = scales_by_tid.get(tid)
        out = model.rollout(I, V, T, soc0=torch.ones(1, device=device), scales=scales)
        yp = out["soc"].detach().cpu().numpy()[0]
        vp = out["voltage"].detach().cpu().numpy()[0]
        resid = out["voltage_residual"].detach().cpu().numpy()[0]
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
            "y_pred": yp,
            "V_raw": f["V_raw"].to_numpy(np.float32),
            "V_pred": vp,
            "voltage_residual": resid,
        })
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = np.abs(df["error"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def adapt_voltage_only(model, frame, cfg: CFG, steps=12, lr=0.03, drift=0.02):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    scales = nn.Parameter(torch.zeros(5, device=device))
    opt = torch.optim.Adam([scales], lr=float(lr))
    f = frame.reset_index(drop=True)
    I_np = f["I_raw"].to_numpy(np.float32)
    V_np = f["V_raw"].to_numpy(np.float32)
    T_np = f["temperature"].to_numpy(np.float32)
    n = len(f)
    chunk = min(int(getattr(cfg, "ecm_tta_chunk_len", 512)), n)
    rng = np.random.default_rng(123)
    for _ in range(int(steps)):
        starts = [0] if n <= chunk else rng.integers(0, n - chunk + 1, size=4).tolist()
        loss = scales.new_tensor(0.0)
        for st in starts:
            sl = slice(st, st + chunk)
            I = torch.as_tensor(I_np[sl][None, :], device=device)
            V = torch.as_tensor(V_np[sl][None, :], device=device)
            T = torch.as_tensor(T_np[sl][None, :], device=device)
            out = model.rollout(I, V, T, soc0=torch.ones(1, device=device), scales=scales)
            loss = loss + F.smooth_l1_loss(out["voltage"], V, beta=0.03)
        loss = loss / len(starts) + float(drift) * torch.mean(scales ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            scales.clamp_(-np.log(2.0), np.log(2.0))
    for p in model.parameters():
        p.requires_grad_(True)
    return scales.detach()


def summarize_voltage_residual(pred):
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for (experiment, model), g in pred.groupby(["experiment", "model_name"]):
        r = g["voltage_residual"].to_numpy(np.float64)
        rows.append({
            "experiment": experiment,
            "model_name": model,
            "voltage_MAE_V": float(np.mean(np.abs(r))),
            "voltage_RMSE_V": float(np.sqrt(np.mean(r ** 2))),
            "voltage_bias_V": float(np.mean(r)),
        })
    return pd.DataFrame(rows)


def attach_and_focus(pred, feature_frames, experiment, omitted_temp):
    if pred.empty:
        return pred, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    lookup = build_prediction_feature_lookup(feature_frames)
    attached = []
    for name, g in pred.groupby("model_name"):
        p = g.assign(split="test", ablation=name)
        attached.append(attach_prediction_features(p, lookup, ablation_name=name, target_label="physical"))
    out = pd.concat(attached, ignore_index=True)
    out["experiment"] = experiment
    overall = _overall_metrics(out)
    overall["experiment"] = experiment
    by_temp = variance_by_temperature(out)
    by_temp["experiment"] = experiment
    focus_rows = []
    for model, g in by_temp.groupby("model_name"):
        omitted = g[np.isclose(g["temperature_C"].astype(float), float(omitted_temp))]
        seen = g[~np.isclose(g["temperature_C"].astype(float), float(omitted_temp))]
        focus_rows.append({
            "experiment": experiment,
            "model_name": model,
            "omitted_temperature_C": float(omitted_temp),
            "omitted_MAE_pct": float(omitted["MAE_pct"].iloc[0]) if len(omitted) else np.nan,
            "omitted_RMSE_pct": float(omitted["RMSE_pct"].iloc[0]) if len(omitted) else np.nan,
            "omitted_jitter_ratio": float(omitted["jitter_ratio"].iloc[0]) if len(omitted) else np.nan,
            "seen_MAE_pct": float(seen["MAE_pct"].mean()) if len(seen) else np.nan,
            "seen_RMSE_pct": float(seen["RMSE_pct"].mean()) if len(seen) else np.nan,
            "overall_MAE_pct": float(overall[overall["model_name"].eq(model)]["MAE_pct"].iloc[0]),
            "overall_RMSE_pct": float(overall[overall["model_name"].eq(model)]["RMSE_pct"].iloc[0]),
            "worst_temperature_MAE_pct": float(g["MAE_pct"].max()) if len(g) else np.nan,
            "temperature_MAE_variance": float(g["MAE_pct"].var()) if len(g) > 1 else np.nan,
        })
    return out, overall, by_temp, pd.DataFrame(focus_rows)


def append_baseline_focus(focus):
    path = Path("rex_omitted_temp_focus.csv")
    if not path.exists() or focus.empty:
        return focus
    base = pd.read_csv(path)
    keep = []
    for exp, g in base.groupby("experiment"):
        if exp not in set(focus["experiment"]):
            continue
        for name in ["R5_GATED"]:
            rows = g[g["model_name"].eq(name)]
            if len(rows):
                keep.append(rows.iloc[[0]])
        aug_rex = g[g["model_name"].str.startswith("R5_GATED_AUG_REX")]
        if len(aug_rex):
            keep.append(aug_rex.sort_values("omitted_MAE_pct").iloc[[0]])
    if not keep:
        return focus
    return pd.concat([pd.concat(keep, ignore_index=True), focus], ignore_index=True)


def run_neural_ecm_experiment(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50"),
    include_tta=True,
):
    configure_torch_runtime()
    cfg = cfg or make_cfg()
    all_pred, all_results, all_by_temp, all_focus = [], [], [], []
    all_vres, all_param, all_history, all_tta = [], [], [], []
    specs = [
        ECMSpec(name="NeuralECMObserver", use_rex=False, lambda_v=0.20, correction_limit=0.05),
        ECMSpec(name="NeuralECMObserver_REX", use_rex=True, lambda_v=0.20, lambda_rex=0.5, lambda_worst=0.10, correction_limit=0.05),
    ]
    for experiment in experiments:
        ecfg = ecm_exp_cfg(cfg, experiment)
        feature_frames = load_filtered_feature_frames(ecfg, decomposed_dir=ecfg.decomposed_dir)
        omitted = omitted_temp_for(experiment)
        pred_rows = []
        rex_model = None
        for spec in specs:
            model, history, _ = train_ecm_model(feature_frames, ecfg, spec, experiment)
            all_history.append(history)
            pred = predict_full_trajectories(model, feature_frames["test"], spec.name)
            pred_rows.append(pred)
            pc = model.parameter_curves()
            pc["experiment"] = experiment
            pc["model_name"] = spec.name
            all_param.append(pc)
            if spec.name == "NeuralECMObserver_REX":
                rex_model = model
        if include_tta and rex_model is not None:
            scales = {}
            tta_scale_rows = []
            for frame in feature_frames["test"]:
                temp = float(frame["temperature"].iloc[0])
                if not np.isclose(temp, omitted):
                    continue
                tid = frame["trajectory_id"].iloc[0]
                s = adapt_voltage_only(rex_model, frame, ecfg, steps=int(getattr(ecfg, "ecm_tta_steps", 12)))
                scales[tid] = s
                tta_scale_rows.append({
                    "experiment": experiment,
                    "trajectory_id": tid,
                    "temperature_C": temp,
                    "log_Q_scale": float(s[0].cpu()),
                    "log_R0_scale": float(s[1].cpu()),
                    "log_RC_gain_scale": float(s[2].cpu()),
                    "log_tau_scale": float(s[3].cpu()),
                    "log_hys_gain_scale": float(s[4].cpu()),
                })
            if scales:
                pred_tta = predict_full_trajectories(
                    rex_model,
                    feature_frames["test"],
                    "NeuralECMObserver_REX_TTA_voltage_only",
                    scales_by_tid=scales,
                )
                pred_rows.append(pred_tta)
                all_tta.append(pd.DataFrame(tta_scale_rows))
        pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        pred["experiment"] = experiment
        all_vres.append(summarize_voltage_residual(pred))
        attached, overall, by_temp, focus = attach_and_focus(pred, feature_frames, experiment, omitted)
        all_pred.append(attached)
        all_results.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
    pred = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    focus_with_baselines = append_baseline_focus(focus)
    vres = pd.concat(all_vres, ignore_index=True) if all_vres else pd.DataFrame()
    params = pd.concat(all_param, ignore_index=True) if all_param else pd.DataFrame()
    history = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    tta = pd.concat(all_tta, ignore_index=True) if all_tta else pd.DataFrame()

    output_dir = cfg.output_dir
    pred.to_csv(output_dir / "neural_ecm_prediction_rows.csv", index=False)
    results.to_csv(output_dir / "neural_ecm_results.csv", index=False)
    by_temp.to_csv(output_dir / "neural_ecm_by_temperature.csv", index=False)
    focus_with_baselines.to_csv(output_dir / "neural_ecm_omitted_temp_focus.csv", index=False)
    focus[focus["experiment"].isin(OUTSIDE_SPECS)].to_csv(output_dir / "neural_ecm_outside_range.csv", index=False)
    vres.to_csv(output_dir / "neural_ecm_voltage_residual.csv", index=False)
    params.to_csv(output_dir / "neural_ecm_parameter_curves.csv", index=False)
    history.to_csv(output_dir / "neural_ecm_training_history.csv", index=False)
    tta.to_csv(output_dir / "neural_ecm_tta_results.csv", index=False)
    print("Neural ECM omitted-temp focus:")
    print(focus_with_baselines.sort_values(["experiment", "omitted_MAE_pct"]).to_string(index=False))
    return {
        "prediction_rows": pred,
        "results": results,
        "by_temperature": by_temp,
        "focus": focus_with_baselines,
        "outside": focus[focus["experiment"].isin(OUTSIDE_SPECS)],
        "voltage_residual": vres,
        "parameter_curves": params,
        "tta": tta,
    }
