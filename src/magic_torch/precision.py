"""Precision and device configuration matching Fortran's precision_mod."""

import os
import torch

from .config import _config

# Device selection: config dict or MAGIC_DEVICE env var overrides auto-detection
_device_override = str(_config.get("device", os.environ.get("MAGIC_DEVICE", ""))).lower()
if _device_override:
    DEVICE = torch.device(_device_override)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# MPS does not support float64/complex128 — use float32/complex64 there
if DEVICE.type == "mps":
    DTYPE = torch.float32
    CDTYPE = torch.complex64
else:
    DTYPE = torch.float64
    CDTYPE = torch.complex128
