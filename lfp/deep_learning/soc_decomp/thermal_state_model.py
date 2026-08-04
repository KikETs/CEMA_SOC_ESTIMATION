from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, make_cfg
from .runtime import configure_torch_runtime, device
from .variance_control import _overall_metrics, variance_by_temperature
from .training import attach_prediction_features, build_prediction_feature_lookup
from .neural_ecm_observer import (
    ECMSpec,
    ECMChunkDataset,
    ECMParameterNet,
    MonotonicOCV,
    adapt_voltage_only,
    balanced_chunk_loader,
    move_batch,
    q_ref_from_frames,
)
from .smoothq_retrain import (
    EXPERIMENTS,
    clone_cfg,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)


@dataclass
class ObserverSpec:
    name: str
    use_rex: bool = False
    lambda_v: float = 0.20
    lambda_rex: float = 0.5
    lambda_worst: float = 0.10
    lambda_param: float = 0.02
    correction_limit: float = 0.05
    init_soc_noise: float = 0.02
    state_noise: float = 0.01
    model_kind: str = "thermal"
    use_tta: bool = False
    use_current_integration: bool = True


class ThermalECMObserver(nn.Module):
    """Neural ECM observer with a learned internal thermal state.

    The thermal state is intentionally low dimensional and bounded. It is not a
    measured cell-core temperature; it is a causal latent state that lets the ECM
    parameter network see drive-cycle heat history in addition to ambient T.
    """

    def __init__(
        self,
        q_ref_ah=1.10,
        dt_sec=1.0,
        correction_limit=0.05,
        thermal_mode="tamb_tcore_heatproxy",
        use_current_integration=True,
    ):
        super().__init__()
        self.dt_sec = float(dt_sec)
        self.correction_limit = float(correction_limit)
        self.thermal_mode = str(thermal_mode)
        self.use_current_integration = bool(use_current_integration)
        self.params = ECMParameterNet(q_ref_ah=q_ref_ah)
        self.ocv = MonotonicOCV()
        self.context_net = nn.Sequential(
            nn.Linear(5, 48),
            nn.SiLU(),
            nn.Linear(48, 8),
        )
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)
        # Softplus-constrained thermal constants. Units are abstract but the
        # update is causal and stable for dt=1s.
        self.raw_C_th = nn.Parameter(torch.tensor(850.0, dtype=torch.float32))
        self.raw_R_th = nn.Parameter(torch.tensor(np.log(np.exp(0.28) - 1.0), dtype=torch.float32))
        self.raw_R_heat_scale = nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1.0), dtype=torch.float32))
        self.core_blend_logit = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _scale_params(self, scales, batch_size, device_):
        if scales is None:
            return None
        s = scales.to(device=device_, dtype=torch.float32)
        if s.ndim == 1:
            s = s.view(1, 5).expand(batch_size, 5)
        return s

    def _thermal_constants(self):
        C_th = 50.0 + F.softplus(self.raw_C_th)
        R_th = 0.03 + F.softplus(self.raw_R_th)
        heat_scale = 0.05 + F.softplus(self.raw_R_heat_scale)
        return C_th, R_th, heat_scale

    def _apply_context(self, p, T_amb, T_core, heat_lp):
        mode = self.thermal_mode
        if mode == "tamb_only":
            T_core = T_amb
            heat_lp = torch.zeros_like(heat_lp)
        elif mode == "tcore_only":
            T_amb = T_core
            heat_lp = torch.zeros_like(heat_lp)
        elif mode == "tamb_tcore":
            heat_lp = torch.zeros_like(heat_lp)
        elif mode == "tamb_heatproxy":
            T_core = T_amb
        elif mode == "tamb_tcore_heatproxy":
            pass
        else:
            raise ValueError(f"Unknown thermal_mode={mode}")
        delta_t = (T_core - T_amb).clamp(-30.0, 60.0)
        context = torch.stack(
            [
                (T_amb - 20.0) / 40.0,
                (T_core - 20.0) / 40.0,
                delta_t / 20.0,
                torch.log1p(heat_lp.clamp_min(0.0)) / 3.0,
                torch.tanh(heat_lp / 2.0),
            ],
            dim=-1,
        )
        raw = self.context_net(context)
        mult = torch.exp(0.20 * torch.tanh(raw))
        keys = ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]
        out = dict(p)
        for i, key in enumerate(keys):
            out[key] = p[key] * mult[..., i]
        return out

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
        T_core = T[:, 0].clone()
        heat_lp = I.new_zeros(B)
        if self.training and state_noise > 0:
            v1 = v1 + torch.randn_like(v1) * float(state_noise)
            v2 = v2 + torch.randn_like(v2) * float(state_noise)
            hys = hys + torch.randn_like(hys) * float(state_noise) * 0.5
            T_core = T_core + torch.randn_like(T_core) * 0.5
        C_th, R_th, heat_scale = self._thermal_constants()
        blend = torch.sigmoid(self.core_blend_logit)
        soc_out, v_out, resid_out, hys_out = [], [], [], []
        tcore_out, heat_out = [], []
        param_trace = {k: [] for k in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]}
        for t in range(L):
            it = I[:, t]
            vt = V[:, t]
            tt = T[:, t]
            if self.thermal_mode == "tamb_only":
                T_eff = tt
            elif self.thermal_mode == "tcore_only":
                T_eff = T_core
            elif self.thermal_mode == "tamb_heatproxy":
                T_eff = tt
            else:
                T_eff = tt + blend * (T_core - tt)
            p0 = self.params(T_eff, soc, it.abs(), scales=scales)
            p = self._apply_context(p0, tt, T_core, heat_lp)
            v_prior = self.ocv(soc, T_eff) + hys - v1 - v2 - it * p["R0"]
            resid = vt - v_prior
            dsoc = (p["k_soc"] * resid).clamp(-self.correction_limit, self.correction_limit)
            soc_c = (soc + dsoc).clamp(0.0, 1.0)
            v1_c = v1 - p["k_v1"] * resid
            v2_c = v2 - p["k_v2"] * resid
            hys_c = (hys + p["k_hys"] * resid).clamp(-0.20, 0.20)
            v_pred = self.ocv(soc_c, T_eff) + hys_c - v1_c - v2_c - it * p["R0"]
            soc_out.append(soc_c.unsqueeze(1))
            v_out.append(v_pred.unsqueeze(1))
            resid_out.append((vt - v_pred).unsqueeze(1))
            hys_out.append(hys_c.unsqueeze(1))
            tcore_out.append(T_core.unsqueeze(1))
            heat_out.append(heat_lp.unsqueeze(1))
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
            heat = (it ** 2) * p["R0"].detach().clamp_min(1e-4) * heat_scale
            heat_lp = 0.995 * heat_lp + 0.005 * heat
            dT = (self.dt_sec / C_th) * (heat - (T_core - tt) / R_th)
            T_core = (T_core + dT).clamp(tt - 20.0, tt + 55.0)
        return {
            "soc": torch.cat(soc_out, dim=1),
            "voltage": torch.cat(v_out, dim=1),
            "voltage_residual": torch.cat(resid_out, dim=1),
            "hys": torch.cat(hys_out, dim=1),
            "T_core": torch.cat(tcore_out, dim=1),
            "heat_proxy": torch.cat(heat_out, dim=1),
            "params": {k: torch.cat(v, dim=1) for k, v in param_trace.items()},
        }

    def regularization(self, device_):
        temps = torch.linspace(-10.0, 50.0, 13, device=device_)
        socs = torch.linspace(0.05, 0.95, 19, device=device_)
        tt, ss = torch.meshgrid(temps, socs, indexing="ij")
        ii = torch.full_like(tt, 0.8)
        p0 = self.params(tt.reshape(-1), ss.reshape(-1), ii.reshape(-1))
        p = self._apply_context(p0, tt.reshape(-1), tt.reshape(-1), torch.zeros_like(tt.reshape(-1)))
        reg = tt.new_tensor(0.0)
        for key in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]:
            val = p[key].reshape(len(temps), len(socs))
            reg = reg + torch.mean((val[1:, :] - val[:-1, :]) ** 2)
            reg = reg + torch.mean((val[:, 1:] - val[:, :-1]) ** 2)
        r0 = p["R0"].reshape(len(temps), len(socs))
        reg = reg + 5.0 * torch.mean(F.relu(r0[1:, :] - r0[:-1, :]) ** 2)
        inc = F.softplus(self.ocv.raw_inc)
        reg = reg + 0.02 * torch.mean((inc[1:] - inc[:-1]) ** 2)
        C_th, R_th, heat_scale = self._thermal_constants()
        reg = reg + 1e-6 * (C_th ** 2) + 1e-3 * (R_th ** 2) + 1e-3 * (heat_scale ** 2)
        reg = reg + 1e-3 * torch.mean(self.context_net[-1].weight ** 2)
        return reg

    def parameter_curves(self):
        rows = []
        param = next(self.parameters())
        with torch.no_grad():
            temps = torch.tensor([-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0], device=param.device)
            socs = torch.linspace(0.05, 0.95, 19, device=param.device)
            C_th, R_th, heat_scale = self._thermal_constants()
            for temp in temps:
                for soc in socs:
                    p0 = self.params(temp.view(1), soc.view(1), torch.tensor([0.8], device=param.device))
                    p = self._apply_context(p0, temp.view(1), temp.view(1), torch.zeros(1, device=param.device))
                    row = {
                        "temperature_C": float(temp.cpu()),
                        "SOC": float(soc.cpu()),
                        "T_core_assumed_C": float(temp.cpu()),
                        "thermal_mode": self.thermal_mode,
                        "C_th": float(C_th.detach().cpu()),
                        "R_th": float(R_th.detach().cpu()),
                        "R_heat_scale": float(heat_scale.detach().cpu()),
                    }
                    for key in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]:
                        row[key] = float(p[key].detach().cpu().item())
                    row["OCV"] = float(self.ocv(soc.view(1), temp.view(1)).detach().cpu().item())
                    rows.append(row)
        return pd.DataFrame(rows)


def observer_batch_loss(model, I, V, T, y, meta, spec: ObserverSpec):
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
    temps_t = temps.to(device=soc_pred.device, dtype=torch.float32) if torch.is_tensor(temps) else torch.as_tensor(temps, device=soc_pred.device, dtype=torch.float32)
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


def train_observer_model(feature_frames, cfg: CFG, spec: ObserverSpec, experiment: str, model_factory):
    chunk_len = int(getattr(cfg, "ecm_chunk_len", getattr(cfg, "window_len", 50)))
    chunk_stride = int(getattr(cfg, "ecm_chunk_stride", getattr(cfg, "stride", 1)))
    train_ds = ECMChunkDataset(feature_frames["train"], chunk_len=chunk_len, stride=chunk_stride)
    if len(train_ds) == 0:
        raise ValueError(f"No train chunks for {spec.name}")
    train_loader = balanced_chunk_loader(train_ds, cfg, shuffle=True)
    model = model_factory(q_ref_from_frames(feature_frames["train"]), float(getattr(cfg, "dt_sec", 1.0)), spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(cfg, "ecm_lr", 8e-4)), weight_decay=1e-4)
    history = []
    temp_loss_rows = []
    epochs = int(getattr(cfg, "ecm_epochs", getattr(cfg, "lstm_epochs", 10)))
    print_every = max(1, int(getattr(cfg, "lstm_print_every", 25)))
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
            loss, parts, by_temp = observer_batch_loss(model, I, V, T, y, meta, spec)
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
            print(f"{experiment} {spec.name} early-stop at epoch={ep}: {row['stop_reason']}")
            break
    return model, pd.DataFrame(history), pd.DataFrame(temp_loss_rows)


@torch.no_grad()
def predict_observer_trajectories(model, frames, model_name, scales_by_tid=None):
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
            "y_pred": out["soc"].detach().cpu().numpy()[0],
            "V_raw": f["V_raw"].to_numpy(np.float32),
            "V_pred": out["voltage"].detach().cpu().numpy()[0],
            "voltage_residual": out["voltage_residual"].detach().cpu().numpy()[0],
        })
        if "T_core" in out:
            df["T_core_est"] = out["T_core"].detach().cpu().numpy()[0]
            df["T_core_minus_amb"] = df["T_core_est"] - df["temperature_C"].astype(float)
        if "heat_proxy" in out:
            df["heat_proxy"] = out["heat_proxy"].detach().cpu().numpy()[0]
        for key, val in out.get("params", {}).items():
            df[key] = val.detach().cpu().numpy()[0]
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = np.abs(df["error"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def attach_and_summarize_observer(pred, feature_frames, experiment, omitted_temp):
    if pred.empty:
        return pred, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    lookup = build_prediction_feature_lookup(feature_frames)
    attached = []
    for name, g in pred.groupby("model_name"):
        p = g.assign(split="test", ablation=name)
        attached.append(attach_prediction_features(p, lookup, ablation_name=name, target_label="physical"))
    out = pd.concat(attached, ignore_index=True)
    out["experiment"] = experiment
    out["label_policy"] = "physical_smoothQ"
    overall = _overall_metrics(out)
    overall["experiment"] = experiment
    overall["label_policy"] = "physical_smoothQ"
    by_temp = variance_by_temperature(out)
    by_temp["experiment"] = experiment
    by_temp["label_policy"] = "physical_smoothQ"
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
            "label_policy": "physical_smoothQ",
        })
    return out, overall, by_temp, pd.DataFrame(focus_rows)


def summarize_voltage_residual(pred):
    rows = []
    if pred.empty or "voltage_residual" not in pred.columns:
        return pd.DataFrame()
    for (experiment, model), g in pred.groupby(["experiment", "model_name"]):
        r = g["voltage_residual"].to_numpy(np.float64)
        rows.append({
            "experiment": experiment,
            "model_name": model,
            "voltage_residual_MAE_V": float(np.mean(np.abs(r))),
            "voltage_residual_RMSE_V": float(np.sqrt(np.mean(r ** 2))),
            "voltage_residual_bias_V": float(np.mean(r)),
            "soc_abs_error_corr": float(np.corrcoef(np.abs(r), g["abs_error"].to_numpy(float))[0, 1])
            if len(g) > 3 and np.std(r) > 0 and np.std(g["abs_error"].to_numpy(float)) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def write_thermal_plots(pred, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if pred.empty or "T_core_est" not in pred.columns:
        return
    for (experiment, model, tid), g in pred.groupby(["experiment", "model_name", "trajectory_id"]):
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        x = np.arange(len(g))
        ax[0].plot(x, g["y_true"], label="true")
        ax[0].plot(x, g["y_pred"], label="pred")
        ax[0].set_ylabel("SOC")
        ax[0].legend(loc="best")
        ax[1].plot(x, g["temperature_C"], label="T_amb")
        ax[1].plot(x, g["T_core_est"], label="T_core_est")
        ax[1].set_ylabel("degC")
        ax[1].legend(loc="best")
        ax[2].plot(x, g["abs_error"] * 100.0, label="abs SOC error")
        ax[2].set_ylabel("error %")
        ax[2].set_xlabel("sample")
        fig.suptitle(f"{experiment} {model} {tid}")
        fig.tight_layout()
        safe = f"{experiment}_{model}_{tid}".replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=140)
        plt.close(fig)


def write_thermal_report(pred, focus, output_dir: Path):
    lines = [
        "# Thermal State Vs Error Report",
        "",
        "The thermal state is a causal latent heat-history state, not a measured core temperature.",
        "All supervised training uses DST/US06 only; FUDS rows are test targets.",
        "",
    ]
    if len(focus):
        lines.extend(["## Omitted/Outside Focus", focus.to_markdown(index=False), ""])
    if len(pred) and "T_core_minus_amb" in pred.columns:
        diag = pred.groupby(["experiment", "model_name", "drive_cycle", "temperature_C"]).agg(
            mean_T_core_minus_amb_C=("T_core_minus_amb", "mean"),
            max_T_core_minus_amb_C=("T_core_minus_amb", "max"),
            MAE_pct=("abs_error", lambda x: float(np.mean(x) * 100.0)),
        ).reset_index()
        diag.to_csv(output_dir / "thermal_state_core_delta_by_cycle.csv", index=False)
        lines.extend([
            "## Thermal State Diagnostic",
            "See `thermal_state_core_delta_by_cycle.csv` for whether the estimated core state differs by cycle at the same ambient temperature.",
            "",
        ])
    lines.extend([
        "## Interpretation Guardrails",
        "- Improvement from `NeuralECM_THERMAL_TTA` is voltage-only adaptation, not pure extrapolation.",
        "- If high-temperature outside-range remains weak, report it directly.",
        "- Do not treat `T_core_est` as a measured physical core temperature.",
    ])
    (output_dir / "thermal_state_vs_error_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_thermal_state_experiment(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50"),
    include_tta=True,
):
    configure_torch_runtime()
    base_cfg = configure_strict_training(clone_cfg(cfg))
    output_dir = base_cfg.output_dir
    lookup = load_smoothq_lookup(output_dir)
    specs = [
        ObserverSpec(name="NeuralECM_THERMAL", use_rex=False, lambda_v=0.20, correction_limit=0.05),
        ObserverSpec(name="NeuralECM_THERMAL_REX", use_rex=True, lambda_v=0.20, lambda_rex=0.5, lambda_worst=0.10, correction_limit=0.05),
    ]
    all_pred, all_results, all_by_temp, all_focus = [], [], [], []
    all_hist, all_temp_loss, all_param, all_tta = [], [], [], []
    for experiment in experiments:
        print(f"=== thermal state experiment {experiment} ===")
        ecfg = experiment_cfg(base_cfg, experiment)
        configure_strict_training(ecfg)
        feature_frames = load_relabelled_frames(ecfg, experiment, lookup)
        omitted = EXPERIMENTS[experiment]["omitted_temp_C"]
        pred_rows = []
        rex_model = None
        for spec in specs:
            model, hist, temp_loss = train_observer_model(
                feature_frames,
                ecfg,
                spec,
                experiment,
                lambda q, dt, s: ThermalECMObserver(q_ref_ah=q, dt_sec=dt, correction_limit=s.correction_limit),
            )
            all_hist.append(hist)
            all_temp_loss.append(temp_loss)
            pred = predict_observer_trajectories(model, feature_frames["test"], spec.name)
            pred_rows.append(pred)
            pc = model.parameter_curves()
            pc["experiment"] = experiment
            pc["model_name"] = spec.name
            all_param.append(pc)
            if spec.name == "NeuralECM_THERMAL_REX":
                rex_model = model
        if include_tta and rex_model is not None:
            scales = {}
            scale_rows = []
            for frame in feature_frames["test"]:
                temp = float(frame["temperature"].iloc[0])
                if not np.isclose(temp, omitted):
                    continue
                tid = frame["trajectory_id"].iloc[0]
                s = adapt_voltage_only(rex_model, frame, ecfg, steps=int(getattr(ecfg, "ecm_tta_steps", 12)))
                scales[tid] = s
                scale_rows.append({
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
                pred_rows.append(
                    predict_observer_trajectories(
                        rex_model,
                        feature_frames["test"],
                        "NeuralECM_THERMAL_REX_TTA_voltage_only",
                        scales_by_tid=scales,
                    )
                )
                all_tta.append(pd.DataFrame(scale_rows))
        pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        pred["experiment"] = experiment
        attached, overall, by_temp, focus = attach_and_summarize_observer(pred, feature_frames, experiment, omitted)
        all_pred.append(attached)
        all_results.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
        pd.concat(all_pred, ignore_index=True).to_csv(output_dir / "thermal_state_prediction_rows.csv", index=False)
        pd.concat(all_results, ignore_index=True).to_csv(output_dir / "thermal_state_results.csv", index=False)
        pd.concat(all_by_temp, ignore_index=True).to_csv(output_dir / "thermal_state_by_temperature.csv", index=False)
        pd.concat(all_focus, ignore_index=True).to_csv(output_dir / "thermal_state_omitted_focus.csv", index=False)
    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    hist = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    temp_loss = pd.concat(all_temp_loss, ignore_index=True) if all_temp_loss else pd.DataFrame()
    params = pd.concat(all_param, ignore_index=True) if all_param else pd.DataFrame()
    tta = pd.concat(all_tta, ignore_index=True) if all_tta else pd.DataFrame()
    vres = summarize_voltage_residual(pred_all)
    pred_all.to_csv(output_dir / "thermal_state_prediction_rows.csv", index=False)
    results.to_csv(output_dir / "thermal_state_results.csv", index=False)
    by_temp.to_csv(output_dir / "thermal_state_by_temperature.csv", index=False)
    focus.to_csv(output_dir / "thermal_state_omitted_focus.csv", index=False)
    hist.to_csv(output_dir / "thermal_state_training_history.csv", index=False)
    temp_loss.to_csv(output_dir / "thermal_state_loss_by_temperature.csv", index=False)
    params.to_csv(output_dir / "thermal_state_parameter_curves.csv", index=False)
    tta.to_csv(output_dir / "thermal_state_tta_scales.csv", index=False)
    vres.to_csv(output_dir / "thermal_state_voltage_residual.csv", index=False)
    write_thermal_plots(pred_all, output_dir / "thermal_state_trajectory_plots")
    write_thermal_report(pred_all, focus, output_dir)
    return {
        "predictions": pred_all,
        "results": results,
        "by_temperature": by_temp,
        "focus": focus,
        "history": hist,
        "parameter_curves": params,
        "tta": tta,
    }


def run_thermal_state_ablation(
    cfg: CFG | None = None,
    *,
    experiments=("Exp A", "Exp B", "Exp C", "Omit N10", "Omit 50"),
):
    configure_torch_runtime()
    base_cfg = configure_strict_training(clone_cfg(cfg))
    output_dir = base_cfg.output_dir
    lookup = load_smoothq_lookup(output_dir)
    specs = [
        ("NeuralECM_Tamb_only", "tamb_only"),
        ("NeuralECM_Tcore_only", "tcore_only"),
        ("NeuralECM_Tamb_Tcore", "tamb_tcore"),
        ("NeuralECM_Tamb_heatproxy", "tamb_heatproxy"),
        ("NeuralECM_Tamb_Tcore_heatproxy", "tamb_tcore_heatproxy"),
    ]
    all_pred, all_results, all_by_temp, all_focus = [], [], [], []
    all_hist, all_param = [], []
    for experiment in experiments:
        print(f"=== thermal state ablation {experiment} ===")
        ecfg = experiment_cfg(base_cfg, experiment)
        configure_strict_training(ecfg)
        feature_frames = load_relabelled_frames(ecfg, experiment, lookup)
        omitted = EXPERIMENTS[experiment]["omitted_temp_C"]
        pred_rows = []
        for name, mode in specs:
            spec = ObserverSpec(name=name, use_rex=False, lambda_v=0.20, correction_limit=0.05)
            model, hist, _ = train_observer_model(
                feature_frames,
                ecfg,
                spec,
                experiment,
                lambda q, dt, s, mode=mode: ThermalECMObserver(
                    q_ref_ah=q,
                    dt_sec=dt,
                    correction_limit=s.correction_limit,
                    thermal_mode=mode,
                ),
            )
            hist["thermal_mode"] = mode
            all_hist.append(hist)
            pred_rows.append(predict_observer_trajectories(model, feature_frames["test"], spec.name))
            pc = model.parameter_curves()
            pc["experiment"] = experiment
            pc["model_name"] = name
            pc["thermal_mode"] = mode
            all_param.append(pc)
        pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        pred["experiment"] = experiment
        attached, overall, by_temp, focus = attach_and_summarize_observer(pred, feature_frames, experiment, omitted)
        all_pred.append(attached)
        all_results.append(overall)
        all_by_temp.append(by_temp)
        all_focus.append(focus)
        pd.concat(all_pred, ignore_index=True).to_csv(output_dir / "thermal_state_ablation_prediction_rows.csv", index=False)
        pd.concat(all_results, ignore_index=True).to_csv(output_dir / "thermal_state_ablation_results.csv", index=False)
        pd.concat(all_by_temp, ignore_index=True).to_csv(output_dir / "thermal_state_ablation_by_temperature.csv", index=False)
        pd.concat(all_focus, ignore_index=True).to_csv(output_dir / "thermal_state_ablation_focus.csv", index=False)
    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    history = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    params = pd.concat(all_param, ignore_index=True) if all_param else pd.DataFrame()
    pred_all.to_csv(output_dir / "thermal_state_ablation_prediction_rows.csv", index=False)
    results.to_csv(output_dir / "thermal_state_ablation_results.csv", index=False)
    by_temp.to_csv(output_dir / "thermal_state_ablation_by_temperature.csv", index=False)
    focus.to_csv(output_dir / "thermal_state_ablation_focus.csv", index=False)
    history.to_csv(output_dir / "thermal_state_ablation_history.csv", index=False)
    params.to_csv(output_dir / "thermal_state_ablation_parameter_curves.csv", index=False)
    summary_rows = []
    if len(focus):
        for scope, df in [
            ("omitted_A_B_C", focus[focus["experiment"].isin(["Exp A", "Exp B", "Exp C"])]),
            ("outside_range", focus[focus["experiment"].isin(["Omit N10", "Omit 50"])]),
        ]:
            for model, g in df.groupby("model_name"):
                summary_rows.append({
                    "model_name": model,
                    "scope": scope,
                    "average_MAE_pct": float(g["omitted_MAE_pct"].mean()),
                    "worst_MAE_pct": float(g["omitted_MAE_pct"].max()),
                    "average_RMSE_pct": float(g["omitted_RMSE_pct"].mean()),
                    "average_jitter_ratio": float(g["omitted_jitter_ratio"].mean()),
                    "n_folds": int(g["experiment"].nunique()),
                })
    summary = pd.DataFrame(summary_rows).sort_values(["scope", "average_MAE_pct"]) if summary_rows else pd.DataFrame()
    summary.to_csv(output_dir / "thermal_state_ablation_summary_table.csv", index=False)
    lines = [
        "# Thermal-State Ablation Summary",
        "",
        "All variants use smoothQ physical SOC labels, DST/US06 train cycles, FUDS test cycle, stride=1, and 300-epoch training with plateau early stop.",
        "",
    ]
    if len(summary):
        lines.extend(["## Aggregate", summary.to_markdown(index=False), ""])
    lines.extend([
        "## Interpretation Rules",
        "- If `Tcore` variants improve over `Tamb_only`, the causal internal thermal state carries useful history beyond ambient temperature.",
        "- If `Tamb_heatproxy` is close to `Tamb_Tcore_heatproxy`, heat-history features approximate the explicit state.",
        "- If `Tamb_only` is weak, ambient temperature alone is insufficient.",
    ])
    (output_dir / "thermal_state_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "predictions": pred_all,
        "results": results,
        "by_temperature": by_temp,
        "focus": focus,
        "summary": summary,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run thermal-state NeuralECM experiments with smoothQ labels.")
    parser.add_argument("--experiments", default="Exp A,Exp B,Exp C,Omit N10,Omit 50")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    exps = tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    if args.ablation:
        out = run_thermal_state_ablation(experiments=exps)
    else:
        out = run_thermal_state_experiment(experiments=exps, include_tta=not args.no_tta)
    print(out["focus"].to_string(index=False))


if __name__ == "__main__":
    main()
