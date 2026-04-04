"""Quick diagnostic: run testOutputs config for 10 steps and check energies."""
import os
import sys

# Set env vars BEFORE importing magic_torch
os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_LMAX"] = "85"
os.environ["MAGIC_NR"] = "73"
os.environ["MAGIC_NCHEBMAX"] = "71"
os.environ["MAGIC_MINC"] = "1"
os.environ["MAGIC_RA"] = "3e5"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_PR"] = "1.0"
os.environ["MAGIC_PRMAG"] = "5.0"
os.environ["MAGIC_RADRATIO"] = "0.35"
os.environ["MAGIC_STRAT"] = "0.1"
os.environ["MAGIC_POLIND"] = "2.0"
os.environ["MAGIC_G0"] = "0.0"
os.environ["MAGIC_G1"] = "1.0"
os.environ["MAGIC_G2"] = "0.0"
os.environ["MAGIC_MODE"] = "0"
os.environ["MAGIC_DTMAX"] = "1e-4"
os.environ["MAGIC_ALPHA"] = "0.6"
os.environ["MAGIC_COURFAC"] = "2.5"
os.environ["MAGIC_ALFFAC"] = "1.0"
os.environ["MAGIC_INIT_S1"] = "404"
os.environ["MAGIC_AMP_S1"] = "0.01"
os.environ["MAGIC_INIT_V1"] = "0"
os.environ["MAGIC_KTOPV"] = "2"
os.environ["MAGIC_KBOTV"] = "2"
os.environ["MAGIC_L_POWER"] = "false"
os.environ["MAGIC_L_HEL"] = "false"
os.environ["MAGIC_L_HEMI"] = "false"
os.environ["MAGIC_N_LOG_STEP"] = "10"

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt, build_all_matrices
from magic_torch.output import get_e_kin, get_e_kin_full
from magic_torch import fields
from magic_torch.params import l_mag, l_anel, n_r_max, l_max, n_cheb_max
from magic_torch.pre_calculations import BuoFac, CorFac, LFfac, eScale
from magic_torch.radial_functions import rho0, rgrav, beta, or1, or2

import torch
import numpy as np

print(f"l_max={l_max}, n_r_max={n_r_max}, n_cheb_max={n_cheb_max}")
print(f"l_anel={l_anel}, l_mag={l_mag}")
print(f"BuoFac={BuoFac:.6e}, CorFac={CorFac:.6e}, LFfac={LFfac:.6e}, eScale={eScale}")
print(f"rho0 range: [{rho0.min():.6e}, {rho0.max():.6e}]")
print(f"rgrav range: [{rgrav.min():.6e}, {rgrav.max():.6e}]")
print(f"beta range: [{beta.min():.6e}, {beta.max():.6e}]")

initialize_fields()
setup_initial_state()

dt = 1e-4
initialize_dt(dt)

# Check initial fields
print(f"\nInitial field norms:")
print(f"  |s|_max = {fields.s_LMloc.abs().max():.6e}")
print(f"  |w|_max = {fields.w_LMloc.abs().max():.6e}")
print(f"  |z|_max = {fields.z_LMloc.abs().max():.6e}")
print(f"  |b|_max = {fields.b_LMloc.abs().max():.6e}")
print(f"  |aj|_max = {fields.aj_LMloc.abs().max():.6e}")

# Check initial energy
ek0 = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
print(f"\nStep 0: e_p={ek0.e_p:.8e}, e_t={ek0.e_t:.8e}")

# Reference values (from reference.out, e_kin section)
ref_ekin = [
    (0.0, 0.0, 0.0),  # t=0
    (41.5802551, 51.7361769, 40.7612105),  # t=1e-3 (step 10)
    (147.800565, 66.1176167, 145.500878),  # t=2e-3 (step 20)
]

# Run 10 steps
sim_time = 0.0
for n in range(1, 11):
    dt_actual = one_step(n, dt)
    sim_time += dt_actual

    if n <= 3 or n == 10:
        ek = get_e_kin_full(fields.w_LMloc, fields.dw_LMloc, fields.z_LMloc)
        print(f"Step {n:3d} (t={sim_time:.6e}): e_p={ek.e_p:.8e}, e_t={ek.e_t:.8e}, "
              f"e_p_as={ek.e_p_as:.8e}, e_t_as={ek.e_t_as:.8e}")

print(f"\nReference at t=1e-3: e_p={ref_ekin[1][0]:.8e}, e_t={ref_ekin[1][1]:.8e}, e_p_as={ref_ekin[1][2]:.8e}")
print(f"Ratio (ref/python) at step 10: e_p={ref_ekin[1][0]/ek.e_p:.2f}x, e_t={ref_ekin[1][1]/ek.e_t:.2f}x")
