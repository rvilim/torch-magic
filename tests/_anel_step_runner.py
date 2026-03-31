"""Anelastic step runner: set env vars, run N steps, dump results."""
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
os.environ["MAGIC_ALPHA"] = "0.6"
os.environ["MAGIC_L_CORRECT_AMZ"] = "true"
os.environ["MAGIC_L_CORRECT_AME"] = "true"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields

# Initialize
initialize_fields()
setup_initial_state()

out_dir = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
dt = 1e-4

initialize_dt(dt)

field_names = ["s", "ds", "p", "dp", "w", "dw", "ddw", "z", "dz"]

for step in range(1, n_steps + 1):
    one_step(step, dt)

    field_map = {
        "s": fields.s_LMloc,
        "ds": fields.ds_LMloc,
        "p": fields.p_LMloc,
        "dp": fields.dp_LMloc,
        "w": fields.w_LMloc,
        "dw": fields.dw_LMloc,
        "ddw": fields.ddw_LMloc,
        "z": fields.z_LMloc,
        "dz": fields.dz_LMloc,
    }

    for name, tensor in field_map.items():
        arr = tensor.cpu().to(torch.complex128).numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)

print(f"Anelastic step runner completed: {n_steps} steps")
