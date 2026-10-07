"""Pick the compute device and its default dtype: Apple GPU (mps), then CUDA, then CPU."""
from __future__ import annotations


def pick_device(torch_mod=None) -> str:
    """'mps' if available, else 'cuda', else 'cpu'. ``torch_mod`` is injectable for tests."""
    if torch_mod is None:
        try:
            import torch as torch_mod  # mac or gpu extra
        except ImportError:
            return "cpu"
    mps = getattr(getattr(torch_mod, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch_mod.cuda.is_available():
        return "cuda"
    return "cpu"


def default_dtype(device: str) -> str:
    """float16 on Apple GPUs, bfloat16 on CUDA, float32 on CPU."""
    return {"mps": "float16", "cuda": "bfloat16"}.get(device, "float32")
