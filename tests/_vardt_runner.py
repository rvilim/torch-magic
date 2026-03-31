"""Variable-dt runner: set env vars, import, run N steps, dump results."""
import os
os.environ["MAGIC_DTMAX"] = "5e-4"
os.environ["MAGIC_INTFAC"] = "10.0"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt, tscheme
from magic_torch.params import dtmax
from magic_torch import fields

N_STEPS = int(sys.argv[2])
out_dir = sys.argv[1]

initialize_fields()
setup_initial_state()
initialize_dt(dtmax)

field_attrs = [
    "w_LMloc", "dw_LMloc", "ddw_LMloc",
    "z_LMloc", "dz_LMloc",
    "s_LMloc", "ds_LMloc",
    "p_LMloc", "dp_LMloc",
    "b_LMloc", "db_LMloc", "ddb_LMloc",
    "aj_LMloc", "dj_LMloc",
]
field_ref_names = [
    "w", "dw", "ddw", "z", "dz", "s", "ds", "p", "dp",
    "b", "db", "ddb", "aj", "dj",
]

dt = dtmax
for step in range(1, N_STEPS + 1):
    dt_new = one_step(n_time_step=step, dt=dt)

    # Save dt values
    np.save(os.path.join(out_dir, f"dt_new_step{step}.npy"),
            np.array(dt_new, dtype=np.float64))
    np.save(os.path.join(out_dir, f"dt1_step{step}.npy"),
            np.array(tscheme.dt[0].item(), dtype=np.float64))
    np.save(os.path.join(out_dir, f"dt2_step{step}.npy"),
            np.array(tscheme.dt[1].item(), dtype=np.float64))

    # Save fields
    for attr, ref_name in zip(field_attrs, field_ref_names):
        arr = getattr(fields, attr).cpu().numpy()
        np.save(os.path.join(out_dir, f"{ref_name}_step{step}.npy"), arr)

    dt = dt_new

print("Variable-dt runner completed successfully")
