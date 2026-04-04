"""Runner for condIC diagnostic tests: e_mag_ic, lorentz_torque, viscous_torque.

Usage: python _condic_diag_runner.py <checkpoint_path> <out_dir>
Env vars: MAGIC_TIME_SCHEME, MAGIC_LMAX, MAGIC_NR, MAGIC_MINC, etc.
"""
import sys
import os
import numpy as np

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

os.environ.setdefault("MAGIC_DEVICE", "cpu")

from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt
from magic_torch import fields
from magic_torch.output import get_e_mag_ic, get_lorentz_torque_ic, get_viscous_torque
from magic_torch import radial_functions as rf
from magic_torch import params

load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(2.0e-4)

# e_mag_ic (conducting path: uses b_ic, db_ic, aj_ic)
emag = get_e_mag_ic(fields.b_LMloc, fields.b_ic, fields.db_ic, fields.aj_ic)
np.save(os.path.join(out_dir, "e_mag_ic.npy"),
        np.array([emag.e_p, emag.e_t, emag.e_p_as, emag.e_t_as]))

# Lorentz torque at ICB
lt = get_lorentz_torque_ic(fields.b_LMloc, fields.db_LMloc, fields.aj_LMloc)
np.save(os.path.join(out_dir, "lorentz_torque_ic.npy"), np.array(lt))

# Viscous torque at ICB (n_r_icb = last OC radial level)
n_r_icb = params.n_r_max - 1
z10 = fields.z_LMloc[1, n_r_icb]  # l=1, m=0
dz10 = fields.dz_LMloc[1, n_r_icb]
vt_icb = get_viscous_torque(
    z10, dz10, rf.r[n_r_icb], rf.beta[n_r_icb], rf.visc[n_r_icb])
np.save(os.path.join(out_dir, "viscous_torque_icb.npy"), np.array(vt_icb))

# Viscous torque at CMB (n_r_cmb = 0)
z10_cmb = fields.z_LMloc[1, 0]
dz10_cmb = fields.dz_LMloc[1, 0]
vt_cmb = get_viscous_torque(
    z10_cmb, dz10_cmb, rf.r[0], rf.beta[0], rf.visc[0])
np.save(os.path.join(out_dir, "viscous_torque_cmb.npy"), np.array(vt_cmb))

print("condIC diag runner done")
