import os

import torch


def select_torch_device(requested=None):
    requested = str(requested or os.environ.get("CEMA_TORCH_DEVICE", "auto")).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CEMA_TORCH_DEVICE=cuda requested, but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("CEMA_TORCH_DEVICE=mps requested, but MPS is unavailable")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("CEMA_TORCH_DEVICE must be auto, cuda, mps, or cpu")
    return torch.device(requested)


device = select_torch_device()


def configure_torch_runtime():
    if device.type == "cuda":
        deterministic = str(os.environ.get("SOC_DETERMINISTIC", "")).strip().lower() in {"1", "true", "yes", "on"}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return device
