from __future__ import annotations

import numpy as np


def propagate_hysteresis(h: float, current_A: float, dt_s: float, q_ref_Ah: float, gamma: float) -> tuple[float, float]:
    """Causal one-state Plett-equivalent hysteresis update.

    Internal current is discharge-positive. The asymptotic state is -1 on
    discharge and +1 on charge, so a positive half-gap raises charge voltage
    and lowers discharge voltage.
    """
    throughput = abs(float(current_A)) * max(float(dt_s), 0.0) / (3600.0 * max(float(q_ref_Ah), 1e-12))
    alpha = float(np.exp(-max(float(gamma), 0.0) * throughput))
    if abs(float(current_A)) < 1e-12:
        return float(np.clip(h, -1.0, 1.0)), alpha
    target = -float(np.sign(current_A))
    updated = alpha * float(h) + (1.0 - alpha) * target
    return float(np.clip(updated, -1.0, 1.0)), alpha


def hysteresis_jacobian_alpha(current_A: float, dt_s: float, q_ref_Ah: float, gamma: float) -> float:
    throughput = abs(float(current_A)) * max(float(dt_s), 0.0) / (3600.0 * max(float(q_ref_Ah), 1e-12))
    return float(np.exp(-max(float(gamma), 0.0) * throughput))

