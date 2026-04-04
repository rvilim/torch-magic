"""Multi-step runner for FD radial scheme.

Usage: python _fd_multistep_runner.py <out_dir> <n_steps>
Env vars (MAGIC_RADIAL_SCHEME=FD, etc.) must be set before import.
"""
import os
import sys
import numpy as np

out_dir = sys.argv[1]
n_steps = int(sys.argv[2])
os.environ.setdefault("MAGIC_DEVICE", "cpu")

from magic_torch import fields as f, init_fields, params
from magic_torch.step_time import one_step, setup_initial_state, initialize_dt

init_fields.initialize_fields()
setup_initial_state()
initialize_dt(params.dtmax)

OC_FIELDS = [
    ("w_LMloc", f.w_LMloc), ("dw_LMloc", f.dw_LMloc), ("ddw_LMloc", f.ddw_LMloc),
    ("z_LMloc", f.z_LMloc), ("dz_LMloc", f.dz_LMloc),
    ("s_LMloc", f.s_LMloc), ("ds_LMloc", f.ds_LMloc),
    ("p_LMloc", f.p_LMloc), ("dp_LMloc", f.dp_LMloc),
    ("b_LMloc", f.b_LMloc), ("db_LMloc", f.db_LMloc), ("ddb_LMloc", f.ddb_LMloc),
    ("aj_LMloc", f.aj_LMloc), ("dj_LMloc", f.dj_LMloc),
]

IC_FIELDS = []
if params.l_cond_ic:
    IC_FIELDS = [
        ("b_ic", f.b_ic), ("db_ic", f.db_ic), ("ddb_ic", f.ddb_ic),
        ("aj_ic", f.aj_ic), ("dj_ic", f.dj_ic), ("ddj_ic", f.ddj_ic),
    ]

for step in range(1, n_steps + 1):
    one_step(step, params.dtmax)
    for name, field in OC_FIELDS + IC_FIELDS:
        np.save(os.path.join(out_dir, f"{name}_step{step}.npy"),
                field.detach().cpu().numpy())

print(f"Completed {n_steps} steps")
