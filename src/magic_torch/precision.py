"""Precision and device configuration matching Fortran's precision_mod."""

import torch

DTYPE = torch.float64
CDTYPE = torch.complex128

# Device selection: use CUDA if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
