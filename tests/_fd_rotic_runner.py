"""Runner for FD + rotating IC field comparison test.

Usage: python _fd_rotic_runner.py <out_dir>
Env vars (MAGIC_RADIAL_SCHEME=FD, MAGIC_NROTIC=1, etc.) must be set before import.

Runs 1 time step with FD radial scheme + rotating IC (insulating, sigma_ratio=0)
and saves all OC fields + omega_ic to the output directory as .npy files.
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

# Save init fields (OC only — no IC B-fields with sigma_ratio=0)
for name, field in [
    ("w_init", f.w_LMloc), ("dw_init", f.dw_LMloc),
    ("z_init", f.z_LMloc), ("dz_init", f.dz_LMloc),
    ("s_init", f.s_LMloc), ("p_init", f.p_LMloc),
    ("b_init", f.b_LMloc), ("db_init", f.db_LMloc),
    ("ddb_init", f.ddb_LMloc),
    ("aj_init", f.aj_LMloc), ("dj_init", f.dj_LMloc),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), field.detach().cpu().numpy())

# Save omega_ic init
np.save(os.path.join(out_dir, "omega_ic_init.npy"), np.array(f.omega_ic))

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

# Save omega_ic step 1
np.save(os.path.join(out_dir, "omega_ic_step1.npy"), np.array(f.omega_ic))

print("FD rotIC step runner done")
