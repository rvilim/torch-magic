"""Multi-step runner for doubleDiffusion."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_RA"] = "4.8e4"
os.environ["MAGIC_RAXI"] = "1.2e5"
os.environ["MAGIC_SC"] = "3.0"
os.environ["MAGIC_PR"] = "0.3"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import numpy as np
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]
n_steps = int(sys.argv[3])

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

OC_FIELDS = ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "xi_LMloc", "dxi_LMloc")

for step in range(1, n_steps + 1):
    one_step(step, 3.0e-4)
    for name in OC_FIELDS:
        arr = getattr(fields, name).cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)

print(f"Completed {n_steps} steps")
