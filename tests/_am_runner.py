"""AM correction runner: verify conservation and correction profiles."""
import os
os.environ["MAGIC_DEVICE"] = "cpu"
os.environ["MAGIC_STRAT"] = "5.0"
os.environ["MAGIC_POLIND"] = "2.0"
os.environ["MAGIC_G0"] = "0.0"
os.environ["MAGIC_G1"] = "0.0"
os.environ["MAGIC_G2"] = "1.0"
os.environ["MAGIC_RA"] = "1.48638035e5"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_PR"] = "1.0"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_INIT_S1"] = "1010"
os.environ["MAGIC_AMP_S1"] = "0.01"
os.environ["MAGIC_KTOPV"] = "1"
os.environ["MAGIC_KBOTV"] = "1"
os.environ["MAGIC_ALPHA"] = "0.6"
os.environ["MAGIC_L_CORRECT_AMZ"] = "true"
os.environ["MAGIC_L_CORRECT_AME"] = "true"

import sys
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch.update_z import get_angular_moment, _am_z_profile, _am_dz_profile, _am_d2z_profile
from magic_torch.pre_calculations import c_moi_oc
from magic_torch.chebyshev import r
from magic_torch.radial_functions import rho0, beta, dbeta
from magic_torch.radial_derivatives import get_ddr
from magic_torch.blocking import st_lm2
from magic_torch import fields

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# Output c_moi_oc
print(f"c_moi_oc = {c_moi_oc}")

# Verify correction profiles analytically
# _am_z_profile should be rho0 * r^2
r2 = r * r
expected_z = rho0 * r2
assert torch.allclose(expected_z, _am_z_profile), "z profile mismatch"

# _am_dz_profile = d/dr(rho0 * r^2) = rho0*(2r + r^2*beta)
expected_dz = rho0 * (2.0 * r + r2 * beta)
assert torch.allclose(expected_dz, _am_dz_profile), "dz profile mismatch"

# _am_d2z_profile = d^2/dr^2(rho0 * r^2) = rho0*(2 + 4*beta*r + dbeta*r^2 + beta^2*r^2)
expected_d2z = rho0 * (2.0 + 4.0 * beta * r + dbeta * r2 + beta * beta * r2)
assert torch.allclose(expected_d2z, _am_d2z_profile), "d2z profile mismatch"

print("profile_z_ok = 1.0")
print("profile_dz_ok = 1.0")
print("profile_d2z_ok = 1.0")

# Initialize
initialize_fields()
setup_initial_state()
initialize_dt(1e-4)

l1m0 = st_lm2[1, 0].item()
l1m1 = st_lm2[1, 1].item()

# Step 0 AM
z10 = fields.z_LMloc[l1m0, :]
z11 = fields.z_LMloc[l1m1, :]
AM_x, AM_y, AM_z = get_angular_moment(z10, z11)
print(f"AM_z_step0 = {AM_z}")
print(f"AM_x_step0 = {AM_x}")

# Run steps with correction
for step in range(1, n_steps + 1):
    one_step(step, 1e-4)

z10 = fields.z_LMloc[l1m0, :]
z11 = fields.z_LMloc[l1m1, :]
AM_x, AM_y, AM_z = get_angular_moment(z10, z11)
print(f"AM_z_step{n_steps} = {AM_z}")
print(f"AM_x_step{n_steps} = {AM_x}")
print(f"AM_y_step{n_steps} = {AM_y}")

# Now run WITHOUT correction to show drift
os.environ["MAGIC_L_CORRECT_AMZ"] = "false"
os.environ["MAGIC_L_CORRECT_AME"] = "false"

# Re-import with new env (subprocess isolation means we need a fresh run)
# Instead, just manually compute what AM would be without correction
# by running more steps with correction disabled at the Python level
from magic_torch.update_z import l_correct_AMz
import magic_torch.update_z as uz
# Temporarily disable
old_amz = uz.l_correct_AMz
old_ame = uz.l_correct_AMe
uz.l_correct_AMz = False
uz.l_correct_AMe = False

# Re-initialize
initialize_fields()
setup_initial_state()
initialize_dt(1e-4)

for step in range(1, n_steps + 1):
    one_step(step, 1e-4)

z10 = fields.z_LMloc[l1m0, :]
z11 = fields.z_LMloc[l1m1, :]
AM_x_u, AM_y_u, AM_z_u = get_angular_moment(z10, z11)
print(f"AM_z_uncorrected_step{n_steps} = {AM_z_u}")

# Restore
uz.l_correct_AMz = old_amz
uz.l_correct_AMe = old_ame
