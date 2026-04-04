"""condIC/condICrotIC multistep runner: take N steps, save fields at each step."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import numpy as np

nrotic = sys.argv[1]
out_dir = sys.argv[2]
n_steps = int(sys.argv[3])

os.environ["MAGIC_NROTIC"] = nrotic

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields
from magic_torch.params import dtmax

initialize_fields()
setup_initial_state()
initialize_dt(dtmax)

OC_FIELDS = ("w_LMloc", "dw_LMloc", "ddw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc")
IC_FIELDS = ("b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic")

for step in range(1, n_steps + 1):
    one_step(step, dtmax)
    for name in OC_FIELDS + IC_FIELDS:
        arr = getattr(fields, name).cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"), arr)
    if int(nrotic) == 1:
        np.save(os.path.join(out_dir, f"omega_ic_step{step}.npy"),
                np.array(fields.omega_ic))

print(f"Completed {n_steps} steps (nRotIC={nrotic})")
