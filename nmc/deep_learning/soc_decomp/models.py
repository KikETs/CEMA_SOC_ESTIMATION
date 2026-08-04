import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import CFG

# VoltageCorrector and diagonal scans
class DiagSelectiveScan(nn.Module):
    def __init__(self, n_state: int, feat_dim: int, dt_init: float = 1.0):
        super().__init__()
        self.n = n_state
        self.feat_dim = feat_dim
        self.a_raw = nn.Parameter(torch.randn(n_state) * 0.02)
        self.b_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, n_state), nn.Softplus()
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, n_state)
        )
        self.d = nn.Parameter(torch.zeros(1))
        self.dt = float(dt_init)

    @torch.no_grad()
    def init_time_constants(self, taus_sec):
        taus = torch.tensor(taus_sec, dtype=torch.float32, device=self.a_raw.device)[: self.n]
        A = -1.0 / taus
        x = (-A).clamp(min=1e-6)
        self.a_raw.copy_(torch.log(torch.exp(x) - 1.0))

    def forward(self, u, feat, x0=None):
        B, L, _ = u.shape
        x = u.new_zeros((B, self.n)) if x0 is None else x0
        A = -F.softplus(self.a_raw).view(1, self.n)
        alpha = torch.exp(A * self.dt).expand(B, self.n)
        flat_feat = feat.reshape(B * L, self.feat_dim)
        b_all = self.b_mlp(flat_feat).reshape(B, L, self.n)
        c_all = self.c_mlp(flat_feat).reshape(B, L, self.n)
        y_out = []
        for t in range(L):
            ut = u[:, t, :]
            b = b_all[:, t, :]
            c = c_all[:, t, :]
            x = alpha * x + (1.0 - alpha) * b * ut
            yt = torch.sum(c * x, dim=-1, keepdim=True) + self.d * ut
            y_out.append(yt)
        return torch.stack(y_out, dim=1), x


class TempTauDiagSelectiveScan(nn.Module):
    """Diagonal scan whose recurrence time constant can vary with learned dynamic inputs."""

    def __init__(
        self,
        n_state: int,
        feat_dim: int,
        dt_init: float = 1.0,
        tau_min_sec: float = 0.25,
        tau_max_sec: float = 4096.0,
        tau_log_scale: float = 1.25,
    ):
        super().__init__()
        self.n = n_state
        self.feat_dim = feat_dim
        self.dt = float(dt_init)
        self.tau_min_sec = float(tau_min_sec)
        self.tau_max_sec = float(tau_max_sec)
        self.tau_log_scale = float(tau_log_scale)
        self.tau_raw = nn.Parameter(torch.zeros(n_state))
        self.tau_delta_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.SiLU(),
            nn.Linear(32, n_state)
        )
        self.b_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, n_state), nn.Softplus()
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, n_state)
        )
        self.d = nn.Parameter(torch.zeros(1))
        self.last_tau_reg = None
        self.last_tau_mean = None
        nn.init.zeros_(self.tau_delta_mlp[-1].weight)
        nn.init.zeros_(self.tau_delta_mlp[-1].bias)

    @torch.no_grad()
    def init_time_constants(self, taus_sec):
        taus = torch.tensor(taus_sec, dtype=torch.float32, device=self.tau_raw.device)[: self.n]
        tau_shifted = (taus - self.tau_min_sec).clamp(min=1e-6)
        inv_softplus = torch.where(
            tau_shifted > 20.0,
            tau_shifted,
            torch.log(torch.expm1(tau_shifted)),
        )
        self.tau_raw.copy_(inv_softplus)

    def _base_tau(self):
        return self.tau_min_sec + F.softplus(self.tau_raw).view(1, self.n)

    def forward(self, u, feat, x0=None):
        B, L, _ = u.shape
        x = u.new_zeros((B, self.n)) if x0 is None else x0
        base_tau = self._base_tau().to(device=u.device, dtype=u.dtype)
        flat_feat = feat.reshape(B * L, self.feat_dim)
        b_all = self.b_mlp(flat_feat).reshape(B, L, self.n)
        c_all = self.c_mlp(flat_feat).reshape(B, L, self.n)
        tau_delta_all = torch.tanh(self.tau_delta_mlp(flat_feat)).reshape(B, L, self.n) * self.tau_log_scale
        tau_all = (base_tau.view(1, 1, self.n) * torch.exp(tau_delta_all)).clamp(
            min=self.tau_min_sec,
            max=self.tau_max_sec,
        )
        alpha_all = torch.exp(-self.dt / tau_all).clamp(min=1e-4, max=0.9999)
        y_out = []
        for t in range(L):
            ut = u[:, t, :]
            b = b_all[:, t, :]
            c = c_all[:, t, :]
            alpha = alpha_all[:, t, :]
            x = alpha * x + (1.0 - alpha) * b * ut
            yt = torch.sum(c * x, dim=-1, keepdim=True) + self.d * ut
            y_out.append(yt)
        self.last_tau_reg = tau_delta_all.square().mean()
        self.last_tau_mean = tau_all.detach().mean()
        return torch.stack(y_out, dim=1), x


class TemperatureShiftTauDiagSelectiveScan(nn.Module):
    """Diagonal scan with a temperature-domain shift factor on the recurrence time constant."""

    def __init__(
        self,
        n_state: int,
        feat_dim: int,
        dt_init: float = 1.0,
        mode: str = "arrhenius",
        tau_min_sec: float = 0.25,
        tau_max_sec: float = 4096.0,
        log_aT_limit: float = 1.6094379124341003,
    ):
        super().__init__()
        self.n = n_state
        self.feat_dim = feat_dim
        self.dt = float(dt_init)
        self.mode = str(mode).lower()
        self.tau_min_sec = float(tau_min_sec)
        self.tau_max_sec = float(tau_max_sec)
        self.log_aT_limit = float(log_aT_limit)
        self.tau_raw = nn.Parameter(torch.zeros(n_state))
        self.arr_a = nn.Parameter(torch.zeros(n_state))
        self.arr_b = nn.Parameter(torch.zeros(n_state))
        self.shift_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, n_state)
        )
        self.b_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, n_state), nn.Softplus()
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, n_state)
        )
        self.d = nn.Parameter(torch.zeros(1))
        self.last_tau_reg = None
        self.last_tau_mean = None
        self.last_shift_smooth = None
        nn.init.zeros_(self.shift_mlp[-1].weight)
        nn.init.zeros_(self.shift_mlp[-1].bias)

    @torch.no_grad()
    def init_time_constants(self, taus_sec):
        taus = torch.tensor(taus_sec, dtype=torch.float32, device=self.tau_raw.device)[: self.n]
        tau_shifted = (taus - self.tau_min_sec).clamp(min=1e-6)
        inv_softplus = torch.where(
            tau_shifted > 20.0,
            tau_shifted,
            torch.log(torch.expm1(tau_shifted)),
        )
        self.tau_raw.copy_(inv_softplus)

    def _base_tau(self):
        return self.tau_min_sec + F.softplus(self.tau_raw).view(1, self.n)

    def _log_aT(self, T_s):
        # T_s is the train-fitted scaled temperature channel. The Arrhenius-like variant
        # constrains the shift to be a smooth monotone function of this temperature coordinate.
        if self.mode == "arrhenius":
            log_at = self.arr_a.view(1, 1, self.n) + self.arr_b.view(1, 1, self.n) * (-T_s)
        elif self.mode == "mlp_bounded":
            log_at = self.shift_mlp(T_s)
        elif self.mode == "hybrid":
            log_at = self.arr_a.view(1, 1, self.n) + self.arr_b.view(1, 1, self.n) * (-T_s)
            log_at = log_at + 0.25 * torch.tanh(self.shift_mlp(T_s))
        else:
            raise ValueError(f"Unknown shift tau mode: {self.mode}")
        return log_at.clamp(min=-self.log_aT_limit, max=self.log_aT_limit)

    def forward(self, u, feat, x0=None):
        B, L, _ = u.shape
        x = u.new_zeros((B, self.n)) if x0 is None else x0
        base_tau = self._base_tau().to(device=u.device, dtype=u.dtype)
        flat_feat = feat.reshape(B * L, self.feat_dim)
        b_all = self.b_mlp(flat_feat).reshape(B, L, self.n)
        c_all = self.c_mlp(flat_feat).reshape(B, L, self.n)
        T_s = feat[..., 2:3]
        log_at = self._log_aT(T_s)
        tau_all = (base_tau.view(1, 1, self.n) * torch.exp(log_at)).clamp(
            min=self.tau_min_sec,
            max=self.tau_max_sec,
        )
        alpha_all = torch.exp(-self.dt / tau_all).clamp(min=1e-4, max=0.9999)
        y_out = []
        for t in range(L):
            ut = u[:, t, :]
            b = b_all[:, t, :]
            c = c_all[:, t, :]
            alpha = alpha_all[:, t, :]
            x = alpha * x + (1.0 - alpha) * b * ut
            yt = torch.sum(c * x, dim=-1, keepdim=True) + self.d * ut
            y_out.append(yt)
        self.last_tau_reg = log_at.square().mean()
        self.last_tau_mean = tau_all.detach().mean()
        if L > 1:
            self.last_shift_smooth = torch.abs(log_at[:, 1:, :] - log_at[:, :-1, :]).mean()
        else:
            self.last_shift_smooth = log_at.new_tensor(0.0)
        return torch.stack(y_out, dim=1), x


class PolarizationScan(nn.Module):
    def __init__(self, n_state: int, feat_dim: int, dt_init: float, scan_cls=DiagSelectiveScan, scan_kwargs=None):
        super().__init__()
        self.scan = scan_cls(n_state, feat_dim, dt_init=dt_init, **(scan_kwargs or {}))

    def forward(self, I_s, feat, state=None):
        v_pol_s, stateN = self.scan(I_s, feat, x0=state)
        return v_pol_s, stateN


class Hysteresis2Scan(nn.Module):
    def __init__(self, n_state: int, feat_dim: int, dt_init: float, scan_cls=DiagSelectiveScan, scan_kwargs=None):
        super().__init__()
        self.plus = scan_cls(n_state, feat_dim, dt_init=dt_init, **(scan_kwargs or {}))
        self.minus = scan_cls(n_state, feat_dim, dt_init=dt_init, **(scan_kwargs or {}))
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, I_s, feat, state=None):
        x_plus0, x_minus0 = (None, None) if state is None else state
        v_p_s, x_plus = self.plus(I_s, feat, x0=x_plus0)
        v_m_s, x_minus = self.minus(I_s, feat, x0=x_minus0)
        g = self.gate(feat)
        v_hys_s = g * v_p_s + (1.0 - g) * v_m_s
        return v_hys_s, (x_plus, x_minus), g


class VoltageCorrector(nn.Module):
    def __init__(self, cfg: CFG, dt_init: float, scan_cls=DiagSelectiveScan, scan_kwargs=None):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = 5  # [V_s or zero, I_s, T_s, dI_s, absI_s]
        self.pol_fast = PolarizationScan(cfg.n_pol_fast, self.feat_dim, dt_init=dt_init, scan_cls=scan_cls, scan_kwargs=scan_kwargs)
        self.pol_mid = PolarizationScan(cfg.n_pol_mid, self.feat_dim, dt_init=dt_init, scan_cls=scan_cls, scan_kwargs=scan_kwargs)
        self.pol_slow = PolarizationScan(cfg.n_pol_slow, self.feat_dim, dt_init=dt_init, scan_cls=scan_cls, scan_kwargs=scan_kwargs)
        self.hys = Hysteresis2Scan(cfg.n_hys, self.feat_dim, dt_init=dt_init, scan_cls=scan_cls, scan_kwargs=scan_kwargs)

        self.g_corr_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

        # R0 MLP intentionally has no trailing Softplus. The bounded value is produced from a logit.
        self.r0_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(),
            nn.Linear(32, 1)
        )

        self.temp_expert_gate = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        self.temp_gain_low = nn.Parameter(torch.tensor([1.10, 1.08, 1.18], dtype=torch.float32))
        self.temp_gain_high = nn.Parameter(torch.tensor([0.96, 0.96, 0.94], dtype=torch.float32))

    def make_feat(self, V_s, I_s, T_s, prev_I_s=None):
        dI = torch.zeros_like(I_s)
        if prev_I_s is not None:
            dI[:, 0:1, :] = I_s[:, 0:1, :] - prev_I_s
        if I_s.size(1) > 1:
            dI[:, 1:, :] = I_s[:, 1:, :] - I_s[:, :-1, :]
        absI = torch.abs(I_s)
        v_feat = V_s if bool(getattr(self.cfg, "corr_use_v_feat", False)) else torch.zeros_like(V_s)
        return torch.cat([v_feat, I_s, T_s, dI, absI], dim=-1)

    def _component_temp_gains(self, T_s):
        if not bool(getattr(self.cfg, "use_corr_temp_expert", True)):
            ones = torch.ones_like(T_s)
            return ones, ones, ones, ones
        gate = self.temp_expert_gate(T_s)
        gmin = float(getattr(self.cfg, "corr_temp_gain_min", 0.75))
        gmax = float(getattr(self.cfg, "corr_temp_gain_max", 1.35))
        low = self.temp_gain_low.view(1, 1, 3).to(device=T_s.device, dtype=T_s.dtype)
        high = self.temp_gain_high.view(1, 1, 3).to(device=T_s.device, dtype=T_s.dtype)
        gains = (gate * low + (1.0 - gate) * high).clamp(min=gmin, max=gmax)
        return gains[..., 0:1], gains[..., 1:2], gains[..., 2:3], gate

    def _temp_tau_regularization(self, like_tensor):
        regs = []
        for scan in [
            self.pol_fast.scan,
            self.pol_mid.scan,
            self.pol_slow.scan,
            self.hys.plus,
            self.hys.minus,
        ]:
            reg = getattr(scan, "last_tau_reg", None)
            if torch.is_tensor(reg):
                regs.append(reg.to(device=like_tensor.device, dtype=like_tensor.dtype))
        if not regs:
            return like_tensor.new_tensor(0.0)
        return torch.stack(regs).mean()

    def forward(self, V_s, I_s, T_s, V_raw, I_raw, vmin, vmax, state=None):
        pol_state0 = None if state is None else state.get("pol", None)
        hys_state0 = None if state is None else state.get("hys", None)
        prev_I_s = None if state is None else state.get("prev_I_s", None)

        pf0 = pm0 = ps0 = None
        if isinstance(pol_state0, (tuple, list)) and len(pol_state0) == 3:
            pf0, pm0, ps0 = pol_state0

        feat = self.make_feat(V_s, I_s, T_s, prev_I_s=prev_I_s)
        v_pf_s, pfN = self.pol_fast(I_s, feat, state=pf0)
        v_pm_s, pmN = self.pol_mid(I_s, feat, state=pm0)
        v_ps_s, psN = self.pol_slow(I_s, feat, state=ps0)
        v_hys_s, hys_stateN, gate = self.hys(I_s, feat, state=hys_state0)

        vscale = (vmax - vmin).clamp(min=1e-12) * 0.5
        v_pf_raw = v_pf_s * vscale
        v_pm_raw = v_pm_s * vscale
        v_ps_raw = v_ps_s * vscale
        v_hys_raw = v_hys_s * vscale

        v_pf_raw = torch.tanh(v_pf_raw / max(1e-6, float(self.cfg.pol_fast_limit_V))) * float(self.cfg.pol_fast_limit_V)
        v_pm_raw = torch.tanh(v_pm_raw / max(1e-6, float(self.cfg.pol_mid_limit_V))) * float(self.cfg.pol_mid_limit_V)
        v_ps_raw = torch.tanh(v_ps_raw / max(1e-6, float(self.cfg.pol_slow_limit_V))) * float(self.cfg.pol_slow_limit_V)

        pol_gain, hys_gain, r0_gain, temp_gate = self._component_temp_gains(T_s)

        v_pol_raw = (v_pf_raw + v_pm_raw + v_ps_raw) * pol_gain
        v_pol_raw = torch.tanh(v_pol_raw / max(1e-6, float(self.cfg.pol_limit_V))) * float(self.cfg.pol_limit_V)
        v_hys_raw = torch.tanh((v_hys_raw * hys_gain) / max(1e-6, float(self.cfg.hys_limit_V))) * float(self.cfg.hys_limit_V)

        dI = feat[..., 3:4]
        r0_in = torch.cat([T_s, torch.abs(I_s), torch.abs(dI)], dim=-1)
        r0_logit = self.r0_mlp(r0_in)
        temp = max(1e-6, float(self.cfg.r0_sigmoid_temp))
        r0_min = float(getattr(self.cfg, "ohm_R0_min", 0.0))
        r0_max = float(self.cfg.ohm_R0_max)
        R0 = r0_min + (r0_max - r0_min) * torch.sigmoid(r0_logit / temp)
        R0 = (R0 * r0_gain).clamp(min=r0_min, max=r0_max)
        v_ohm_raw = R0 * I_raw

        g_corr = float(getattr(self.cfg, "g_corr_scale", 1.0)) * self.g_corr_mlp(feat)
        v_drop_raw = g_corr * (v_pol_raw + v_hys_raw + v_ohm_raw)

        if bool(getattr(self.cfg, "enforce_vcorr_floor", True)):
            v_floor = float(getattr(self.cfg, "v_floor_raw", 2.0))
            V_corr_pre = V_raw + v_drop_raw
            if bool(getattr(self.cfg, "use_soft_vfloor", True)):
                beta = float(getattr(self.cfg, "vfloor_softplus_beta", 12.0))
                V_corr_raw = v_floor + F.softplus((V_corr_pre - v_floor) * beta) / max(beta, 1e-6)
            else:
                V_corr_raw = V_corr_pre.clamp(min=v_floor)
            v_drop_capped = V_corr_raw - V_raw
        else:
            V_corr_raw = V_raw + v_drop_raw
            v_drop_capped = v_drop_raw

        vr = (vmax - vmin).clamp(min=1e-12)
        V_corr_s = 2.0 * (V_corr_raw - vmin) / vr - 1.0
        V_corr_s = V_corr_s.clamp(-1.0, 1.0)

        next_state = {
            "pol": (pfN, pmN, psN),
            "hys": hys_stateN,
            "prev_I_s": I_s[:, -1:, :].detach(),
        }
        aux = {
            "feat": feat,
            "gate": gate,
            "R0": R0,
            "v_pol_fast_raw": v_pf_raw,
            "v_pol_mid_raw": v_pm_raw,
            "v_pol_slow_raw": v_ps_raw,
            "v_pol_raw": v_pol_raw,
            "v_hys_raw": v_hys_raw,
            "v_ohm_raw": v_ohm_raw,
            "v_drop_raw": v_drop_raw,
            "v_drop_capped": v_drop_capped,
            "V_raw": V_raw,
            "V_corr_raw": V_corr_raw,
            "V_corr_s": V_corr_s,
            "g_corr": g_corr,
            "corr_temp_gate": temp_gate,
            "temp_tau_reg": self._temp_tau_regularization(V_raw),
        }
        return V_corr_s, next_state, aux


class CorrectorTempTau(VoltageCorrector):
    def __init__(self, cfg: CFG, dt_init: float):
        scan_kwargs = {
            "tau_min_sec": float(getattr(cfg, "temp_tau_min_sec", 0.25)),
            "tau_max_sec": float(getattr(cfg, "temp_tau_max_sec", 4096.0)),
            "tau_log_scale": float(getattr(cfg, "temp_tau_log_scale", 1.25)),
        }
        super().__init__(cfg, dt_init=dt_init, scan_cls=TempTauDiagSelectiveScan, scan_kwargs=scan_kwargs)


class CorrectorSmoothDecomp(VoltageCorrector):
    """VoltageCorrector trained with SOC-friendly smooth decomposition regularizers."""

    pass


class CorrectorSmoothDecompTempTau(CorrectorTempTau):
    """TempTau corrector trained with SOC-friendly smooth decomposition regularizers."""

    pass


class CorrectorShiftTau(VoltageCorrector):
    def __init__(self, cfg: CFG, dt_init: float, mode: str):
        scan_kwargs = {
            "mode": mode,
            "tau_min_sec": float(getattr(cfg, "temp_tau_min_sec", 0.25)),
            "tau_max_sec": float(getattr(cfg, "temp_tau_max_sec", 4096.0)),
            "log_aT_limit": float(getattr(cfg, "shift_tau_log_aT_limit", 1.6094379124341003)),
        }
        super().__init__(cfg, dt_init=dt_init, scan_cls=TemperatureShiftTauDiagSelectiveScan, scan_kwargs=scan_kwargs)


class CorrectorShiftTauArrhenius(CorrectorShiftTau):
    def __init__(self, cfg: CFG, dt_init: float):
        super().__init__(cfg, dt_init=dt_init, mode="arrhenius")


class CorrectorShiftTauMLPBounded(CorrectorShiftTau):
    def __init__(self, cfg: CFG, dt_init: float):
        super().__init__(cfg, dt_init=dt_init, mode="mlp_bounded")


class CorrectorShiftTauHybrid(CorrectorShiftTau):
    def __init__(self, cfg: CFG, dt_init: float):
        super().__init__(cfg, dt_init=dt_init, mode="hybrid")

# Stateless LSTM window dataset and model
class DecomposedWindowDataset(Dataset):
    def __init__(self, frames, feature_cols, window_len, stride, target_label="physical"):
        self.frames = []
        self.feature_cols = list(feature_cols)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.target_label = str(target_label)
        self.index = []
        target_col = self._target_column()
        for fi, frame in enumerate(frames):
            frame = frame.reset_index(drop=True)
            frame_cache = {
                "x": np.ascontiguousarray(frame[self.feature_cols].to_numpy(np.float32)),
                "y_physical": np.ascontiguousarray(frame["SOC_physical"].to_numpy(np.float32)),
                "y_usable": np.ascontiguousarray(frame["SOC_usable_cutoff"].to_numpy(np.float32)),
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
                if target_col is None:
                    is_finite = np.isfinite(frame_cache["y_physical"][end]) and np.isfinite(frame_cache["y_usable"][end])
                else:
                    y_key = "y_physical" if target_col == "SOC_physical" else "y_usable"
                    is_finite = np.isfinite(frame_cache[y_key][end])
                if is_finite:
                    self.index.append((fi, start, end))

    def _target_column(self):
        if self.target_label == "physical":
            return "SOC_physical"
        if self.target_label == "usable":
            return "SOC_usable_cutoff"
        if self.target_label == "multi":
            return None
        raise ValueError(f"unknown target_label={self.target_label}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start, end = self.index[idx]
        f = self.frames[fi]
        x = f["x"][start:end + 1]
        if self.target_label == "multi":
            y = np.array([f["y_physical"][end], f["y_usable"][end]], dtype=np.float32)
        elif self.target_label == "physical":
            y = f["y_physical"][end:end + 1]
        else:
            y = f["y_usable"][end:end + 1]
        soc_for_bin = float(f["y_physical"][end])
        if soc_for_bin < 0.2:
            soc_bin = 0
        elif soc_for_bin <= 0.8:
            soc_bin = 1
        else:
            soc_bin = 2
        meta = {
            "file_name": f["file_name"][end],
            "trajectory_id": f["trajectory_id"][end],
            "end_index": int(f["end_index"][end]),
            "temperature": float(f["temperature"][end]),
            "drive_cycle": f["drive_cycle"][end],
            "soc_bin": int(soc_bin),
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


class StatelessLSTM_SOC(nn.Module):
    def __init__(self, input_dim, hidden_size=64, output_dim=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_dim)

    def encode_last(self, x):
        z = self.input_proj(x)
        out, _ = self.lstm(z)  # No external hidden/cell state: every window resets state.
        out = self.norm(out + z)
        return out[:, -1, :]

    def forward_logits(self, x):
        return self.head(self.encode_last(x))

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))


class FeatureGatedStatelessLSTM_SOC(StatelessLSTM_SOC):
    def __init__(self, input_dim, feature_cols, hidden_size=64, output_dim=1, num_layers=1, dropout=0.0):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.feature_cols = list(feature_cols)
        self.component_names = ["V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        missing_components = [c for c in self.component_names if c not in self.feature_cols]
        if missing_components:
            raise ValueError(f"Gated model requires component features: {missing_components}")
        gate_input_names = ["T", "V_raw", "V_corr_raw", "absI", "dI", "R0", "V_pol_raw", "V_hys_raw", "V_ohm_raw"]
        self.gate_input_names = [c for c in gate_input_names if c in self.feature_cols]
        if not self.gate_input_names:
            raise ValueError("Gated model requires at least one gate input feature")
        self.component_indices = [self.feature_cols.index(c) for c in self.component_names]
        self.gate_input_indices = [self.feature_cols.index(c) for c in self.gate_input_names]
        self.component_gate = nn.Sequential(
            nn.Linear(len(self.gate_input_indices), 32), nn.SiLU(),
            nn.Linear(32, 3), nn.Sigmoid()
        )

    def component_gates(self, x):
        return self.component_gate(x[..., self.gate_input_indices])

    def apply_component_gates(self, x):
        gates = self.component_gates(x)
        gated_by_index = {idx: j for j, idx in enumerate(self.component_indices)}
        cols = []
        for idx in range(x.size(-1)):
            col = x[..., idx:idx + 1]
            if idx in gated_by_index:
                col = col * gates[..., gated_by_index[idx]:gated_by_index[idx] + 1]
            cols.append(col)
        return torch.cat(cols, dim=-1), gates

    def forward_logits(self, x):
        x_eff, _ = self.apply_component_gates(x)
        return self.head(self.encode_last(x_eff))


class TempHeadStatelessLSTM_SOC(StatelessLSTM_SOC):
    def __init__(self, input_dim, feature_cols, hidden_size=64, output_dim=1, num_layers=1, dropout=0.0):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.feature_cols = list(feature_cols)
        if "T" not in self.feature_cols:
            raise ValueError("Temperature residual head requires T in feature_cols")
        self.temp_idx = self.feature_cols.index("T")
        self.temp_head = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, output_dim)
        )
        nn.init.zeros_(self.temp_head[-1].weight)
        nn.init.zeros_(self.temp_head[-1].bias)

    def forward_logits(self, x):
        base_logits = self.head(self.encode_last(x))
        temp_last = x[:, -1, self.temp_idx:self.temp_idx + 1]
        return base_logits + self.temp_head(temp_last)


class GatedTempHeadStatelessLSTM_SOC(FeatureGatedStatelessLSTM_SOC):
    def __init__(self, input_dim, feature_cols, hidden_size=64, output_dim=1, num_layers=1, dropout=0.0):
        super().__init__(
            input_dim=input_dim,
            feature_cols=feature_cols,
            hidden_size=hidden_size,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        if "T" not in self.feature_cols:
            raise ValueError("Temperature residual head requires T in feature_cols")
        self.temp_idx = self.feature_cols.index("T")
        self.temp_head = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(),
            nn.Linear(16, output_dim)
        )
        nn.init.zeros_(self.temp_head[-1].weight)
        nn.init.zeros_(self.temp_head[-1].bias)

    def forward_logits(self, x):
        x_eff, _ = self.apply_component_gates(x)
        base_logits = self.head(self.encode_last(x_eff))
        temp_last = x[:, -1, self.temp_idx:self.temp_idx + 1]
        return base_logits + self.temp_head(temp_last)


def assert_stateless_lstm(model):
    import inspect
    sig = inspect.signature(model.forward)
    assert list(sig.parameters.keys()) == ["x"], "StatelessLSTM_SOC.forward must accept only x"
    assert model.lstm.batch_first is True
    print("stateless LSTM check passed: no hidden/cell input, state resets per window")


def collate_meta_to_frame(meta):
    out = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy().tolist()
        else:
            out[k] = list(v)
    return pd.DataFrame(out)


def build_lstm_soc_model(feature_cols, output_dim, cfg: CFG, ablation_name: str):
    name = str(ablation_name).upper()
    common = {
        "input_dim": len(feature_cols),
        "hidden_size": cfg.lstm_hidden_size,
        "output_dim": output_dim,
        "num_layers": cfg.lstm_layers,
        "dropout": cfg.lstm_dropout,
    }
    if "GATED" in name and "TEMP_HEAD" in name:
        return GatedTempHeadStatelessLSTM_SOC(feature_cols=feature_cols, **common)
    if "GATED" in name:
        return FeatureGatedStatelessLSTM_SOC(feature_cols=feature_cols, **common)
    if "TEMP_HEAD" in name:
        return TempHeadStatelessLSTM_SOC(feature_cols=feature_cols, **common)
    return StatelessLSTM_SOC(**common)


def build_corrector(cfg: CFG, device, dt_init=None):
    variant = str(getattr(cfg, "corrector_variant", "base")).lower()
    if variant in {"base", "voltagecorrector"}:
        corrector = VoltageCorrector(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    elif variant in {"temp_tau", "correctortemptau", "temperature_tau"}:
        corrector = CorrectorTempTau(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    elif variant in {"smooth", "smooth_decomp", "correctorsmoothdecomp"}:
        corrector = CorrectorSmoothDecomp(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    elif variant in {"smooth_temp_tau", "correctorsmoothdecomp_temptau", "smooth_decomp_temp_tau"}:
        corrector = CorrectorSmoothDecompTempTau(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    elif variant in {"shift_tau_arrhenius", "correctorshifttau_arrhenius"}:
        corrector = CorrectorShiftTauArrhenius(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    elif variant in {"shift_tau_mlp_bounded", "shift_tau_mlp", "correctorshifttau_mlpbounded"}:
        corrector = CorrectorShiftTauMLPBounded(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    elif variant in {"shift_tau_hybrid", "correctorshifttau_hybrid"}:
        corrector = CorrectorShiftTauHybrid(cfg, dt_init=cfg.dt_sec if dt_init is None else dt_init).to(device)
    else:
        raise ValueError(f"Unknown corrector_variant={getattr(cfg, 'corrector_variant', None)!r}")
    corrector.pol_fast.scan.init_time_constants([1, 2, 4, 8])
    corrector.pol_mid.scan.init_time_constants([16, 32, 64, 128])
    corrector.pol_slow.scan.init_time_constants([256, 512, 1024, 2048])
    corrector.hys.plus.init_time_constants([2, 4, 8, 16, 32, 64])
    corrector.hys.minus.init_time_constants([2, 4, 8, 16, 32, 64])
    return corrector
