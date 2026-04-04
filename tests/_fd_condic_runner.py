"""Runner for FD + condIC field comparison test.

Usage: python _fd_condic_runner.py <out_dir>
Env vars (MAGIC_RADIAL_SCHEME=FD, MAGIC_SIGMA_RATIO=1, etc.) must be set before import.

Runs 1 time step with FD radial scheme + conducting inner core and saves
all OC + IC fields to the output directory as .npy files.
"""
import os
import sys
import numpy as np

out_dir = sys.argv[1]
os.environ.setdefault("MAGIC_DEVICE", "cpu")

from magic_torch import fields as f, init_fields, params
from magic_torch.step_time import one_step, setup_initial_state, initialize_dt

# Initialize
init_fields.initialize_fields()
setup_initial_state()
initialize_dt(params.dtmax)

# Save init fields (OC + IC)
for name, field in [
    ("w_init", f.w_LMloc), ("dw_init", f.dw_LMloc),
    ("z_init", f.z_LMloc), ("dz_init", f.dz_LMloc),
    ("s_init", f.s_LMloc), ("p_init", f.p_LMloc),
    ("b_init", f.b_LMloc), ("db_init", f.db_LMloc),
    ("ddb_init", f.ddb_LMloc),
    ("aj_init", f.aj_LMloc), ("dj_init", f.dj_LMloc),
    ("b_ic_init", f.b_ic), ("db_ic_init", f.db_ic),
    ("ddb_ic_init", f.ddb_ic),
    ("aj_ic_init", f.aj_ic), ("dj_ic_init", f.dj_ic),
    ("ddj_ic_init", f.ddj_ic),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), field.detach().cpu().numpy())

# Run 1 step
one_step(1, params.dtmax)

# Save step 1 fields (OC)
for name, field in [
    ("w_step1", f.w_LMloc), ("dw_step1", f.dw_LMloc),
    ("ddw_step1", f.ddw_LMloc),
    ("z_step1", f.z_LMloc), ("dz_step1", f.dz_LMloc),
    ("s_step1", f.s_LMloc), ("ds_step1", f.ds_LMloc),
    ("p_step1", f.p_LMloc), ("dp_step1", f.dp_LMloc),
    ("b_step1", f.b_LMloc), ("db_step1", f.db_LMloc),
    ("ddb_step1", f.ddb_LMloc),
    ("aj_step1", f.aj_LMloc), ("dj_step1", f.dj_LMloc),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), field.detach().cpu().numpy())

# Save step 1 IC fields
for name, field in [
    ("b_ic_step1", f.b_ic), ("db_ic_step1", f.db_ic),
    ("ddb_ic_step1", f.ddb_ic),
    ("aj_ic_step1", f.aj_ic), ("dj_ic_step1", f.dj_ic),
    ("ddj_ic_step1", f.ddj_ic),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), field.detach().cpu().numpy())

print("FD condIC step runner done")
