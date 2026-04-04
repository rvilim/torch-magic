"""Runner for anelastic + composition field comparison test.

Usage: python _anel_comp_runner.py <out_dir> [n_steps]
Runs N steps with anelastic (strat=5) + composition (raxi=1e5) and saves fields.
"""
import os
import sys
import numpy as np

os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_STRAT"] = "5.0"
os.environ["MAGIC_POLIND"] = "2.0"
os.environ["MAGIC_G0"] = "0.0"
os.environ["MAGIC_G1"] = "0.0"
os.environ["MAGIC_G2"] = "1.0"
os.environ["MAGIC_RA"] = "1.48638035e5"
os.environ["MAGIC_RAXI"] = "1e5"
os.environ["MAGIC_SC"] = "3.0"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_PR"] = "1.0"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_INIT_S1"] = "1010"
os.environ["MAGIC_AMP_S1"] = "0.01"
os.environ["MAGIC_KTOPV"] = "1"
os.environ["MAGIC_KBOTV"] = "1"
os.environ["MAGIC_ALPHA"] = "0.6"
os.environ["MAGIC_L_CORRECT_AMZ"] = "true"
os.environ["MAGIC_L_CORRECT_AME"] = "true"
os.environ["MAGIC_NCHEBMAX"] = "31"

import torch
from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields as f

out_dir = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

initialize_fields()
setup_initial_state()
dt = 1e-4
initialize_dt(dt)

# Fields to save (mode=1: no magnetic)
_FIELDS = [
    ("s", f.s_LMloc), ("ds", f.ds_LMloc),
    ("p", f.p_LMloc), ("dp", f.dp_LMloc),
    ("w", f.w_LMloc), ("dw", f.dw_LMloc), ("ddw", f.ddw_LMloc),
    ("z", f.z_LMloc), ("dz", f.dz_LMloc),
    ("xi", f.xi_LMloc), ("dxi", f.dxi_LMloc),
]

# Save init
for name, field in _FIELDS:
    np.save(os.path.join(out_dir, f"{name}_init.npy"),
            field.detach().cpu().to(torch.complex128).numpy())

# Run steps
for step in range(1, n_steps + 1):
    one_step(step, dt)
    for name, field in _FIELDS:
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"),
                field.detach().cpu().to(torch.complex128).numpy())

print(f"Anel+comp runner done: {n_steps} steps")
