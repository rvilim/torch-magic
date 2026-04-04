"""Couette flow (mode=7, l_SRIC) test runner.

Runs in subprocess to ensure env vars are set before module import.

Usage:
    python _couette_runner.py <n_steps> <outdir>

Dumps velocity fields at each step to <outdir>/<field>_step<N>.npy.
"""
import os
os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_MODE"] = "7"
os.environ["MAGIC_NROTIC"] = "-1"
os.environ["MAGIC_OMEGA_IC1"] = "-4000.0"
os.environ["MAGIC_RA"] = "0.0"
# Don't set MAGIC_PRMAG=0 — opm=1/prmag would divide by zero
# prmag defaults to 5.0, unused when l_mag=False (mode=7)
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_RADRATIO"] = "0.35"
os.environ["MAGIC_KTOPV"] = "2"
os.environ["MAGIC_KBOTV"] = "2"
os.environ["MAGIC_ALPHA"] = "0.6"
os.environ["MAGIC_DTMAX"] = "1e-4"
os.environ["MAGIC_INIT_S1"] = "0"
os.environ["MAGIC_INIT_V1"] = "0"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt, build_all_matrices
from magic_torch import fields
from magic_torch.params import dtmax

n_steps = int(sys.argv[1])
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

FIELD_NAMES = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "p_LMloc", "dp_LMloc",
    "s_LMloc", "ds_LMloc",
]


def dump_fields(step):
    for name in FIELD_NAMES:
        arr = getattr(fields, name).cpu().to(torch.complex128).numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)
    np.save(os.path.join(out_dir, f"omega_ic_step{step}.npy"),
            np.array(fields.omega_ic))


initialize_fields()
setup_initial_state()
dt = dtmax
initialize_dt(dt)

for step in range(1, n_steps + 1):
    one_step(step, dt)
    dump_fields(step)

print(f"Couette run: {n_steps} steps done, omega_ic={fields.omega_ic}")
