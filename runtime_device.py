from __future__ import annotations

import os

import torch

_WARNED_MESSAGES: set[str] = set()


def resolve_torch_device() -> torch.device:
    pref = os.getenv("CHEMEAGLE_DEVICE", "auto").strip().lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warn_once("CHEMEAGLE_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if pref in {"metal", "mps"}:
        if mps_available:
            return torch.device("mps")
        warn_once("CHEMEAGLE_DEVICE=metal requested but MPS is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")

    return torch.device("cpu")


def easyocr_uses_acceleration(device: torch.device) -> bool:
    return device.type in {"cuda", "mps"}


def warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(message)
