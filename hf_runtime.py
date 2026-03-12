from __future__ import annotations

import os


def configure_transformers_runtime() -> None:
    # ChemEagle uses the PyTorch side of HuggingFace. Disabling TF avoids
    # optional Keras/TensorFlow imports that can break local runs.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


configure_transformers_runtime()
