"""Bisection test: run 5 steps at l_max=64, minc=4 with CNAB2 vs BPR353.
Compare energy drift to determine if bug is in time scheme or spatial operators.

Usage: python _bisect_scheme_runner.py <checkpoint_path> <time_scheme> <out_dir>
"""
import os
import sys

scheme = sys.argv[2]

# Set env vars BEFORE importing magic_torch
os.environ["MAGIC_TIME_SCHEME"] = scheme
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

import numpy as np
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.output import get_e_kin, get_e_mag
from magic_torch import fields

ckpt_path = sys.argv[1]
out_dir = sys.argv[3]
dt = 2.0e-4

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(dt)

energies = []

def log_energy(step, t):
    e_kin_pol, e_kin_tor = get_e_kin(
        fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
        fields.z_LMloc, fields.dz_LMloc)
    e_mag_pol, e_mag_tor = get_e_mag(
        fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
        fields.aj_LMloc, fields.dj_LMloc)
    energies.append([step, t, e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor])

# Log initial energy
log_energy(0, sim_time)

# Run 5 steps, log every step
for n in range(1, 6):
    one_step(n, dt)
    sim_time += dt
    log_energy(n, sim_time)

energies = np.array(energies)
np.save(os.path.join(out_dir, "energies.npy"), energies)
print(f"{scheme} runner completed")
print(f"Step 0 ekin_pol={energies[0,2]:.10e} ekin_tor={energies[0,3]:.10e}")
print(f"Step 5 ekin_pol={energies[5,2]:.10e} ekin_tor={energies[5,3]:.10e}")
drift_pol = (energies[5,2] - energies[0,2]) / energies[0,2] * 100
drift_tor = (energies[5,3] - energies[0,3]) / energies[0,3] * 100
print(f"Drift: ekin_pol={drift_pol:.4f}% ekin_tor={drift_tor:.4f}%")
