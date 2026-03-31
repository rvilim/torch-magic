"""Anelastic profiles runner: dump radial profiles and scalars to tmpdir."""
import os
os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_STRAT"] = "5.0"
os.environ["MAGIC_POLIND"] = "2.0"
os.environ["MAGIC_G0"] = "0.0"
os.environ["MAGIC_G1"] = "0.0"
os.environ["MAGIC_G2"] = "1.0"
os.environ["MAGIC_RA"] = "1.48638035e5"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_PR"] = "1.0"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_INIT_S1"] = "1010"
os.environ["MAGIC_AMP_S1"] = "0.01"
os.environ["MAGIC_KTOPV"] = "1"
os.environ["MAGIC_KBOTV"] = "1"

import sys
import numpy as np
import torch

from magic_torch import radial_functions as rf

out_dir = sys.argv[1]

# Scalars
for name, val in [
    ("DissNb", rf.DissNb),
    ("GrunNb", rf.GrunNb),
    ("ThExpNb", rf.ThExpNb),
    ("ViscHeatFac", rf.ViscHeatFac),
    ("OhmLossFac", rf.OhmLossFac),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), np.array(val, dtype=np.float64))

# Profiles
for name, arr in [
    ("temp0", rf.temp0_f64),
    ("rho0", rf.rho0_f64),
    ("beta", rf.beta_f64),
    ("dbeta", rf.dbeta_f64),
    ("ddbeta", rf.ddbeta_f64),
    ("rgrav", rf.rgrav_f64),
    ("orho1", rf.orho1_f64),
    ("alpha0", rf.alpha0_f64),
    ("ogrun", rf.ogrun_f64),
    ("otemp1", rf.otemp1_f64),
    ("dLtemp0", rf.dLtemp0_f64),
    ("ddLtemp0", rf.ddLtemp0_f64),
]:
    if isinstance(arr, torch.Tensor):
        data = arr.cpu().to(torch.float64).numpy()
    else:
        data = np.array(arr, dtype=np.float64)
    np.save(os.path.join(out_dir, f"{name}.npy"), data)

# Transport
for name, arr in [
    ("visc", rf.visc),
    ("dLvisc", rf.dLvisc),
    ("ddLvisc", rf.ddLvisc),
    ("kappa", rf.kappa),
    ("dLkappa", rf.dLkappa),
]:
    if isinstance(arr, torch.Tensor):
        data = arr.cpu().to(torch.float64).numpy()
    else:
        data = np.array(arr, dtype=np.float64)
    np.save(os.path.join(out_dir, f"{name}.npy"), data)

print("Anel profiles runner completed")
