"""BoussBenchSat runner: load Fortran checkpoint, run 25 steps, dump energies."""
import os
import sys
import numpy as np

# Set env vars BEFORE importing magic_torch
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

from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.output import get_e_kin, get_e_mag
from magic_torch import fields
from magic_torch.params import dtmax

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]
dt = 2.0e-4

# Load Fortran checkpoint
sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(dt)

# Collect energies at step 0
energies = []

def log_energy(step, t):
    e_kin_pol, e_kin_tor = get_e_kin(
        fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
        fields.z_LMloc, fields.dz_LMloc)
    e_mag_pol, e_mag_tor = get_e_mag(
        fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
        fields.aj_LMloc, fields.dj_LMloc)
    energies.append([step, t, e_kin_pol, e_kin_tor, e_mag_pol, e_mag_tor])

# Run 25 steps, log every 5
for n in range(1, 26):
    one_step(n, dt)
    sim_time += dt
    if n % 5 == 0:
        log_energy(n, sim_time)

# Save results
energies = np.array(energies)
np.save(os.path.join(out_dir, "energies.npy"), energies)
np.save(os.path.join(out_dir, "omega_ic.npy"), np.array(fields.omega_ic))
print("BoussBenchSat runner completed")
