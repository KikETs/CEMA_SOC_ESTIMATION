from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .runtime import configure_torch_runtime, device
from .neural_ecm_observer import MonotonicOCV, adapt_voltage_only
from .smoothq_retrain import (
    EXPERIMENTS,
    clone_cfg,
    configure_strict_training,
    experiment_cfg,
    load_relabelled_frames,
    load_smoothq_lookup,
)
from .thermal_state_model import (
    ObserverSpec,
    attach_and_summarize_observer,
    predict_observer_trajectories,
    summarize_voltage_residual,
    train_observer_model,
)


def _logit_from_sigmoid_value(v):
    v = float(np.clip(v, 1e-4, 1.0 - 1e-4))
    return np.log(v / (1.0 - v))


class SurfaceParameterNet(nn.Module):
    """Smooth T/SOC parameter tables with a small bounded neural residual."""

    def __init__(self, q_ref_ah=1.10, residual_limit=0.15):
        super().__init__()
        self.q_ref_ah = float(q_ref_ah)
        self.residual_limit = float(residual_limit)
        t_knots = torch.tensor([-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0], dtype=torch.float32)
        s_knots = torch.linspace(0.0, 1.0, 21, dtype=torch.float32)
        self.register_buffer("t_knots", t_knots)
        self.register_buffer("s_knots", s_knots)
        shape = (len(t_knots), len(s_knots))

        def table(init):
            return nn.Parameter(torch.full(shape, float(init), dtype=torch.float32))

        self.raw_q = table(_logit_from_sigmoid_value((1.0 - 0.45) / 1.20))
        self.raw_r0 = table(_logit_from_sigmoid_value((0.080 - 0.003) / 0.220))
        self.raw_r1 = table(_logit_from_sigmoid_value((0.035 - 0.001) / 0.180))
        self.raw_tau1 = table(_logit_from_sigmoid_value((70.0 - 1.0) / 249.0))
        self.raw_r2 = table(_logit_from_sigmoid_value((0.025 - 0.001) / 0.180))
        self.raw_tau2 = table(_logit_from_sigmoid_value((650.0 - 25.0) / 1975.0))
        self.raw_hys_gain = table(_logit_from_sigmoid_value((0.030 - 0.005) / 0.095))
        self.raw_hys_tau = table(_logit_from_sigmoid_value((700.0 - 50.0) / 2950.0))
        self.residual_net = nn.Sequential(
            nn.Linear(3, 48),
            nn.SiLU(),
            nn.Linear(48, 8),
        )
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)
        self.control_net = nn.Sequential(
            nn.Linear(3, 48),
            nn.SiLU(),
            nn.Linear(48, 5),
        )

    def _features(self, temp_c, soc, abs_i):
        return torch.stack(
            [
                (temp_c - 20.0) / 40.0,
                soc.clamp(0.0, 1.0) * 2.0 - 1.0,
                (abs_i / 4.0).clamp(0.0, 2.0),
            ],
            dim=-1,
        )

    def _interp(self, table, temp_c, soc):
        t = temp_c.clamp(float(self.t_knots[0]), float(self.t_knots[-1]))
        s = soc.clamp(float(self.s_knots[0]), float(self.s_knots[-1]))
        ti1 = torch.bucketize(t.contiguous(), self.t_knots).clamp(1, len(self.t_knots) - 1)
        si1 = torch.bucketize(s.contiguous(), self.s_knots).clamp(1, len(self.s_knots) - 1)
        ti0 = ti1 - 1
        si0 = si1 - 1
        t0 = self.t_knots[ti0]
        t1 = self.t_knots[ti1]
        s0 = self.s_knots[si0]
        s1 = self.s_knots[si1]
        wt = ((t - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0.0, 1.0)
        ws = ((s - s0) / (s1 - s0).clamp_min(1e-6)).clamp(0.0, 1.0)
        v00 = table[ti0, si0]
        v01 = table[ti0, si1]
        v10 = table[ti1, si0]
        v11 = table[ti1, si1]
        return (
            (1 - wt) * (1 - ws) * v00
            + (1 - wt) * ws * v01
            + wt * (1 - ws) * v10
            + wt * ws * v11
        )

    def _positive_params(self, temp_c, soc):
        q = self.q_ref_ah * (0.45 + 1.20 * torch.sigmoid(self._interp(self.raw_q, temp_c, soc)))
        r0 = 0.003 + 0.220 * torch.sigmoid(self._interp(self.raw_r0, temp_c, soc))
        r1 = 0.001 + 0.180 * torch.sigmoid(self._interp(self.raw_r1, temp_c, soc))
        tau1 = 1.0 + 249.0 * torch.sigmoid(self._interp(self.raw_tau1, temp_c, soc))
        r2 = 0.001 + 0.180 * torch.sigmoid(self._interp(self.raw_r2, temp_c, soc))
        tau2 = 25.0 + 1975.0 * torch.sigmoid(self._interp(self.raw_tau2, temp_c, soc))
        hys_gain = 0.005 + 0.095 * torch.sigmoid(self._interp(self.raw_hys_gain, temp_c, soc))
        hys_tau = 50.0 + 2950.0 * torch.sigmoid(self._interp(self.raw_hys_tau, temp_c, soc))
        return q, r0, r1, tau1, r2, tau2, hys_gain, hys_tau

    def forward(self, temp_c, soc, abs_i, scales=None):
        feat = self._features(temp_c, soc, abs_i)
        q, r0, r1, tau1, r2, tau2, hys_gain, hys_tau = self._positive_params(temp_c, soc)
        residual = torch.exp(self.residual_limit * torch.tanh(self.residual_net(feat)))
        q = q * residual[..., 0]
        r0 = r0 * residual[..., 1]
        r1 = r1 * residual[..., 2]
        tau1 = tau1 * residual[..., 3]
        r2 = r2 * residual[..., 4]
        tau2 = tau2 * residual[..., 5]
        hys_gain = hys_gain * residual[..., 6]
        hys_tau = hys_tau * residual[..., 7]
        raw = self.control_net(feat)
        eta = 0.94 + 0.12 * torch.sigmoid(raw[..., 0])
        k_soc = 0.006 * torch.sigmoid(raw[..., 1])
        k_v1 = 0.035 * torch.sigmoid(raw[..., 2])
        k_v2 = 0.025 * torch.sigmoid(raw[..., 3])
        k_hys = 0.020 * torch.sigmoid(raw[..., 4])
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

    def regularization(self):
        reg = self.raw_q.new_tensor(0.0)
        tables = [
            self.raw_q,
            self.raw_r0,
            self.raw_r1,
            self.raw_tau1,
            self.raw_r2,
            self.raw_tau2,
            self.raw_hys_gain,
            self.raw_hys_tau,
        ]
        for tab in tables:
            reg = reg + torch.mean((tab[1:, :] - tab[:-1, :]) ** 2)
            reg = reg + torch.mean((tab[:, 1:] - tab[:, :-1]) ** 2)
        # R0 mostly decreases as temperature increases.
        r0 = 0.003 + 0.220 * torch.sigmoid(self.raw_r0)
        reg = reg + 5.0 * torch.mean(F.relu(r0[1:, :] - r0[:-1, :]) ** 2)
        reg = reg + 1e-3 * torch.mean(self.residual_net[-1].weight ** 2)
        return reg

    def residual_magnitude(self):
        rows = []
        with torch.no_grad():
            temps = self.t_knots
            socs = torch.linspace(0.05, 0.95, 19, device=temps.device)
            for temp in temps:
                for soc in socs:
                    feat = self._features(temp.view(1), soc.view(1), torch.tensor([0.8], device=temps.device))
                    res = torch.exp(self.residual_limit * torch.tanh(self.residual_net(feat)))[0]
                    row = {"temperature_C": float(temp.cpu()), "SOC": float(soc.cpu())}
                    for i, key in enumerate(["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]):
                        row[f"{key}_residual_multiplier"] = float(res[i].detach().cpu())
                    rows.append(row)
        return pd.DataFrame(rows)


class ParamSurfaceECMObserver(nn.Module):
    def __init__(
        self,
        q_ref_ah=1.10,
        dt_sec=1.0,
        correction_limit=0.05,
        residual_limit=0.15,
        use_thermal=False,
        use_current_integration=True,
    ):
        super().__init__()
        self.dt_sec = float(dt_sec)
        self.correction_limit = float(correction_limit)
        self.use_thermal = bool(use_thermal)
        self.use_current_integration = bool(use_current_integration)
        self.params = SurfaceParameterNet(q_ref_ah=q_ref_ah, residual_limit=residual_limit)
        self.ocv = MonotonicOCV()
        if self.use_thermal:
            self.raw_C_th = nn.Parameter(torch.tensor(850.0, dtype=torch.float32))
            self.raw_R_th = nn.Parameter(torch.tensor(np.log(np.exp(0.28) - 1.0), dtype=torch.float32))
            self.raw_R_heat_scale = nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1.0), dtype=torch.float32))

    def _scale_params(self, scales, batch_size, device_):
        if scales is None:
            return None
        s = scales.to(device=device_, dtype=torch.float32)
        if s.ndim == 1:
            s = s.view(1, 5).expand(batch_size, 5)
        return s

    def _thermal_constants(self):
        if not self.use_thermal:
            return None
        return (
            50.0 + F.softplus(self.raw_C_th),
            0.03 + F.softplus(self.raw_R_th),
            0.05 + F.softplus(self.raw_R_heat_scale),
        )

    def rollout(self, I, V, T, soc0=None, train_perturb=0.0, state_noise=0.0, scales=None):
        B, L = I.shape
        scales = self._scale_params(scales, B, I.device)
        soc = I.new_ones(B) if soc0 is None else soc0.to(device=I.device, dtype=torch.float32).clamp(0.0, 1.0)
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
        thermal = self._thermal_constants()
        soc_out, v_out, resid_out, hys_out = [], [], [], []
        tcore_out, heat_out = [], []
        param_trace = {k: [] for k in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]}
        for t in range(L):
            it = I[:, t]
            vt = V[:, t]
            tt = T[:, t]
            T_eff = T_core if self.use_thermal else tt
            p = self.params(T_eff, soc, it.abs(), scales=scales)
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
            if self.use_thermal:
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
            if self.use_thermal and thermal is not None:
                C_th, R_th, heat_scale = thermal
                heat = (it ** 2) * p["R0"].detach().clamp_min(1e-4) * heat_scale
                heat_lp = 0.995 * heat_lp + 0.005 * heat
                T_core = (T_core + (self.dt_sec / C_th) * (heat - (T_core - tt) / R_th)).clamp(tt - 20.0, tt + 55.0)
        out = {
            "soc": torch.cat(soc_out, dim=1),
            "voltage": torch.cat(v_out, dim=1),
            "voltage_residual": torch.cat(resid_out, dim=1),
            "hys": torch.cat(hys_out, dim=1),
            "params": {k: torch.cat(v, dim=1) for k, v in param_trace.items()},
        }
        if self.use_thermal:
            out["T_core"] = torch.cat(tcore_out, dim=1)
            out["heat_proxy"] = torch.cat(heat_out, dim=1)
        return out

    def regularization(self, device_):
        reg = self.params.regularization()
        inc = F.softplus(self.ocv.raw_inc)
        reg = reg + 0.02 * torch.mean((inc[1:] - inc[:-1]) ** 2)
        if self.use_thermal:
            C_th, R_th, heat_scale = self._thermal_constants()
            reg = reg + 1e-6 * (C_th ** 2) + 1e-3 * (R_th ** 2) + 1e-3 * (heat_scale ** 2)
        return reg

    def parameter_curves(self):
        rows = []
        param = next(self.parameters())
        with torch.no_grad():
            temps = torch.tensor([-10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0], device=param.device)
            socs = torch.linspace(0.05, 0.95, 19, device=param.device)
            for temp in temps:
                for soc in socs:
                    p = self.params(temp.view(1), soc.view(1), torch.tensor([0.8], device=param.device))
                    row = {
                        "temperature_C": float(temp.cpu()),
                        "SOC": float(soc.cpu()),
                        "use_thermal": self.use_thermal,
                    }
                    for key in ["Q_eff", "R0", "R1", "tau1", "R2", "tau2", "hys_gain", "hys_tau"]:
                        row[key] = float(p[key].detach().cpu().item())
                    row["OCV"] = float(self.ocv(soc.view(1), temp.view(1)).detach().cpu().item())
                    rows.append(row)
        return pd.DataFrame(rows)

    def residual_magnitude(self):
        out = self.params.residual_magnitude()
        out["use_thermal"] = self.use_thermal
        return out


def write_parameter_surface_plots(curves: pd.DataFrame, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if curves.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    keys = ["Q_eff", "R0", "tau1", "tau2", "hys_gain", "OCV"]
    for ax, key in zip(axes.ravel(), keys):
        for (model, soc), g in curves.groupby(["model_name", "SOC"]):
            if not np.isclose(float(soc), 0.5, atol=0.03):
                continue
            ax.plot(g["temperature_C"], g[key], marker="o", label=model)
        ax.set_title(key)
        ax.set_xlabel("T (C)")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_surface_curves.png", dpi=150)
    plt.close(fig)


def run_parameter_surface_experiment(
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
        ObserverSpec(name="ParamSurfaceECM_REX", use_rex=True, lambda_v=0.20, lambda_rex=0.5, lambda_worst=0.10, correction_limit=0.05),
        ObserverSpec(name="ParamSurfaceECM_THERMAL_REX", use_rex=True, lambda_v=0.20, lambda_rex=0.5, lambda_worst=0.10, correction_limit=0.05),
    ]
    all_pred, all_results, all_by_temp, all_focus = [], [], [], []
    all_hist, all_temp_loss, all_param, all_resid, all_tta = [], [], [], [], []
    for experiment in experiments:
        print(f"=== parameter surface experiment {experiment} ===")
        ecfg = experiment_cfg(base_cfg, experiment)
        configure_strict_training(ecfg)
        feature_frames = load_relabelled_frames(ecfg, experiment, lookup)
        omitted = EXPERIMENTS[experiment]["omitted_temp_C"]
        pred_rows = []
        tta_model = None
        for spec in specs:
            use_thermal = "THERMAL" in spec.name
            model, hist, temp_loss = train_observer_model(
                feature_frames,
                ecfg,
                spec,
                experiment,
                lambda q, dt, s, use_thermal=use_thermal: ParamSurfaceECMObserver(
                    q_ref_ah=q,
                    dt_sec=dt,
                    correction_limit=s.correction_limit,
                    residual_limit=0.15,
                    use_thermal=use_thermal,
                ),
            )
            all_hist.append(hist)
            all_temp_loss.append(temp_loss)
            pred = predict_observer_trajectories(model, feature_frames["test"], spec.name)
            pred_rows.append(pred)
            pc = model.parameter_curves()
            pc["experiment"] = experiment
            pc["model_name"] = spec.name
            all_param.append(pc)
            rm = model.residual_magnitude()
            rm["experiment"] = experiment
            rm["model_name"] = spec.name
            all_resid.append(rm)
            if spec.name == "ParamSurfaceECM_THERMAL_REX":
                tta_model = model
        if include_tta and tta_model is not None:
            scales = {}
            scale_rows = []
            for frame in feature_frames["test"]:
                temp = float(frame["temperature"].iloc[0])
                if not np.isclose(temp, omitted):
                    continue
                tid = frame["trajectory_id"].iloc[0]
                s = adapt_voltage_only(tta_model, frame, ecfg, steps=int(getattr(ecfg, "ecm_tta_steps", 12)))
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
                        tta_model,
                        feature_frames["test"],
                        "ParamSurfaceECM_THERMAL_REX_TTA_voltage_only",
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
        pd.concat(all_pred, ignore_index=True).to_csv(output_dir / "parameter_surface_prediction_rows.csv", index=False)
        pd.concat(all_results, ignore_index=True).to_csv(output_dir / "parameter_surface_results.csv", index=False)
        pd.concat(all_by_temp, ignore_index=True).to_csv(output_dir / "parameter_surface_by_temperature.csv", index=False)
        pd.concat(all_focus, ignore_index=True).to_csv(output_dir / "parameter_surface_omitted_focus.csv", index=False)
    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    by_temp = pd.concat(all_by_temp, ignore_index=True) if all_by_temp else pd.DataFrame()
    focus = pd.concat(all_focus, ignore_index=True) if all_focus else pd.DataFrame()
    hist = pd.concat(all_hist, ignore_index=True) if all_hist else pd.DataFrame()
    temp_loss = pd.concat(all_temp_loss, ignore_index=True) if all_temp_loss else pd.DataFrame()
    curves = pd.concat(all_param, ignore_index=True) if all_param else pd.DataFrame()
    resid = pd.concat(all_resid, ignore_index=True) if all_resid else pd.DataFrame()
    tta = pd.concat(all_tta, ignore_index=True) if all_tta else pd.DataFrame()
    vres = summarize_voltage_residual(pred_all)
    pred_all.to_csv(output_dir / "parameter_surface_prediction_rows.csv", index=False)
    results.to_csv(output_dir / "parameter_surface_results.csv", index=False)
    by_temp.to_csv(output_dir / "parameter_surface_by_temperature.csv", index=False)
    focus.to_csv(output_dir / "parameter_surface_omitted_focus.csv", index=False)
    hist.to_csv(output_dir / "parameter_surface_training_history.csv", index=False)
    temp_loss.to_csv(output_dir / "parameter_surface_loss_by_temperature.csv", index=False)
    curves.to_csv(output_dir / "parameter_surface_curves.csv", index=False)
    resid.to_csv(output_dir / "parameter_surface_residual_magnitude.csv", index=False)
    tta.to_csv(output_dir / "parameter_tta_results.csv", index=False)
    tta.to_csv(output_dir / "parameter_tta_prefix_results.csv", index=False)
    tta.to_csv(output_dir / "parameter_tta_by_target.csv", index=False)
    vres.to_csv(output_dir / "parameter_surface_voltage_residual.csv", index=False)
    write_parameter_surface_plots(curves, output_dir)
    return {
        "predictions": pred_all,
        "results": results,
        "by_temperature": by_temp,
        "focus": focus,
        "history": hist,
        "parameter_curves": curves,
        "residual_magnitude": resid,
        "tta": tta,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter-surface ECM experiments with smoothQ labels.")
    parser.add_argument("--experiments", default="Exp A,Exp B,Exp C,Omit N10,Omit 50")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    exps = tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    out = run_parameter_surface_experiment(experiments=exps, include_tta=not args.no_tta)
    print(out["focus"].to_string(index=False))


if __name__ == "__main__":
    main()
