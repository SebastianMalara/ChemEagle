from __future__ import annotations

import os

_WARNED_MESSAGES: set[str] = set()


def _load_torch():
    import torch

    return torch


def resolve_torch_device():
    torch = _load_torch()
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


def easyocr_uses_acceleration(device) -> bool:
    return getattr(device, "type", "") in {"cuda", "mps"}


def resolve_ocr_backend(
    requested_backend: str | None = None,
    run_mode: str | None = None,
) -> str:
    backend = (requested_backend or os.getenv("OCR_BACKEND") or "auto").strip().lower()
    mode = (run_mode or os.getenv("CHEMEAGLE_RUN_MODE") or "cloud").strip().lower()

    aliases = {
        "vision": "llm_vision",
        "llm": "llm_vision",
        "easy_ocr": "easyocr",
    }
    backend = aliases.get(backend, backend)

    if backend == "auto":
        return "llm_vision" if mode == "cloud" else "easyocr"
    return backend


def warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(message)
