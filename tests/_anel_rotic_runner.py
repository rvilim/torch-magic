"""Runner for anelastic + rotating IC field comparison test.

Anelastic MHD (strat=5, mode=0) with rotating IC (nRotIC=1, kbotv=2, sigma_ratio=0).
Tests z10Mat anelastic profiles and domega_ic_dt.impl beta fix.
"""
import os
import sys
import numpy as np

os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_STRAT"] = "5.0"
os.environ["MAGIC_POLIND"] = "2.0"
os.environ["MAGIC_G0"] = "0.0"
os.environ["MAGIC_G1"] = "0.0"
os.environ["MAGIC_G2"] = "1.0"
os.environ["MAGIC_RA"] = "1.48638035e5"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_PR"] = "1.0"
os.environ["MAGIC_PRMAG"] = "5.0"
os.environ["MAGIC_MODE"] = "0"
os.environ["MAGIC_INIT_S1"] = "404"
os.environ["MAGIC_AMP_S1"] = "0.1"
os.environ["MAGIC_KTOPV"] = "2"
os.environ["MAGIC_KBOTV"] = "2"
os.environ["MAGIC_NROTIC"] = "1"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_ALPHA"] = "0.6"

import torch
from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import fields as f

out_dir = sys.argv[1]

initialize_fields()
setup_initial_state()
initialize_dt(1e-4)

# Save init fields (OC + omega_ic)
for name, field in [
    ("w_init", f.w_LMloc), ("dw_init", f.dw_LMloc),
    ("z_init", f.z_LMloc), ("dz_init", f.dz_LMloc),
    ("s_init", f.s_LMloc), ("p_init", f.p_LMloc),
    ("b_init", f.b_LMloc), ("db_init", f.db_LMloc),
    ("ddb_init", f.ddb_LMloc),
    ("aj_init", f.aj_LMloc), ("dj_init", f.dj_LMloc),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), field.detach().cpu().to(torch.complex128).numpy())
np.save(os.path.join(out_dir, "omega_ic_init.npy"), np.array(f.omega_ic))

# Run 1 step
one_step(1, 1e-4)

# Save step 1 fields
for name, field in [
    ("w_step1", f.w_LMloc), ("dw_step1", f.dw_LMloc), ("ddw_step1", f.ddw_LMloc),
    ("z_step1", f.z_LMloc), ("dz_step1", f.dz_LMloc),
    ("s_step1", f.s_LMloc), ("ds_step1", f.ds_LMloc),
    ("p_step1", f.p_LMloc), ("dp_step1", f.dp_LMloc),
    ("b_step1", f.b_LMloc), ("db_step1", f.db_LMloc), ("ddb_step1", f.ddb_LMloc),
    ("aj_step1", f.aj_LMloc), ("dj_step1", f.dj_LMloc),
]:
    np.save(os.path.join(out_dir, f"{name}.npy"), field.detach().cpu().to(torch.complex128).numpy())
np.save(os.path.join(out_dir, "omega_ic_step1.npy"), np.array(f.omega_ic))

print("Anel+rotIC runner done")
