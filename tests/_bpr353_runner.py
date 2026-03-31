"""BPR353 runner: set env vars, import, run 1 step, dump results."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.params import dtmax
from magic_torch import fields

# Initialize and run 1 step
initialize_fields()
setup_initial_state()
initialize_dt(dtmax)
one_step(n_time_step=1, dt=dtmax)

# Save all 14 fields
out_dir = sys.argv[1]
field_map = {
    "w_step1": fields.w_LMloc,
    "dw_step1": fields.dw_LMloc,
    "ddw_step1": fields.ddw_LMloc,
    "z_step1": fields.z_LMloc,
    "dz_step1": fields.dz_LMloc,
    "s_step1": fields.s_LMloc,
    "ds_step1": fields.ds_LMloc,
    "p_step1": fields.p_LMloc,
    "dp_step1": fields.dp_LMloc,
    "b_step1": fields.b_LMloc,
    "db_step1": fields.db_LMloc,
    "ddb_step1": fields.ddb_LMloc,
    "aj_step1": fields.aj_LMloc,
    "dj_step1": fields.dj_LMloc,
}

for name, tensor in field_map.items():
    arr = tensor.cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

print("BPR353 runner completed successfully")
