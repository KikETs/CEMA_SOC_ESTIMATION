import json
import platform
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn


MEASURED_STEPS = 100_000
WARMUP_STEPS = 1_000


class ProposedG4ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17 * 4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.07),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.unused_normal_head = nn.Linear(128, 1)
        self.anchor = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.07),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.07),
            nn.Linear(64, 1),
        )
        self.residual_limit = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        summary = torch.cat(
            [x[:, -1], x.mean(1), x.std(1, unbiased=False), x[:, -1] - x[:, 0]],
            dim=1,
        )
        encoded = self.encoder(summary)
        anchor = torch.sigmoid(self.anchor(x[:, :, :8]))
        residual = self.residual_limit * torch.tanh(self.residual(encoded))
        return (anchor[:, -1] + residual).clamp(0.0, 1.0)


def cpu_model():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = ProposedG4ResidualMLP().eval()
inputs = torch.zeros((1, 50, 17), dtype=torch.float32)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
if parameter_count != 44_036:
    raise RuntimeError(f"Unexpected proposed parameter count: {parameter_count}")

with torch.inference_mode():
    for _ in range(WARMUP_STEPS):
        model(inputs)
    latency_ns = np.empty(MEASURED_STEPS, dtype=np.int64)
    checksum = 0.0
    for index in range(MEASURED_STEPS):
        started = perf_counter_ns()
        output = model(inputs)
        latency_ns[index] = perf_counter_ns() - started
        checksum += float(output[0, 0])

print(
    json.dumps(
        {
            "method": "proposed (EMA+NN)",
            "benchmark_steps": MEASURED_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "median_latency_us": float(np.median(latency_ns) / 1000.0),
            "p95_latency_us": float(np.percentile(latency_ns, 95) / 1000.0),
            "parameter_count": parameter_count,
            "parameter_bytes_FP32": parameter_count * 4,
            "macs_per_window": 901_888,
            "cpu": cpu_model(),
            "torch_version": torch.__version__,
            "checksum": checksum,
            "scope": "batch-1 NN forward on a precomputed 50x17 causal EMA window; feature generation excluded to match results/complexity.csv",
        }
    )
)
