"""Multi-step runner for boussBenchSat."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_NCHEBICMAX"] = "15"
os.environ["MAGIC_RA"] = "1.1e5"
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"
os.environ["MAGIC_NROTIC"] = "1"
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
initialize_dt(2.0e-4)

OC_FIELDS = ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc")
IC_FIELDS = ("b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic")

for step in range(1, n_steps + 1):
    one_step(step, 2.0e-4)
    for name in OC_FIELDS + IC_FIELDS:
        arr = getattr(fields, name).cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)
    np.save(os.path.join(out_dir, f"omega_ic_step{step}.npy"),
            np.array(fields.omega_ic))

print(f"Completed {n_steps} steps")
