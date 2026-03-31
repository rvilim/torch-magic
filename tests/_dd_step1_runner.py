"""Step1 runner for doubleDiffusion."""
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

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Take one time step
one_step(1, 3.0e-4)

# Dump all OC fields (no magnetic for mode=1)
for name in ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "xi_LMloc", "dxi_LMloc"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

print(f"DD Step1 runner completed, sim_time={sim_time}")
