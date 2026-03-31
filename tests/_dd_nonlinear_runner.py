"""DD nonlinear runner: capture intermediates at step 1 and step 2."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_RA"] = "4.8e4"
os.environ["MAGIC_RAXI"] = "1.2e5"
os.environ["MAGIC_SC"] = "3.0"
os.environ["MAGIC_PR"] = "0.3"
os.environ["MAGIC_EK"] = "1e-3"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import numpy as np
import torch
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import (
    setup_initial_state, initialize_dt, one_step,
    radial_loop, lm_loop, build_all_matrices, tscheme,
)
from magic_torch import fields, dt_fields
from magic_torch.sht import scal_to_spat, torpol_to_spat, torpol_to_curl_spat
from magic_torch.radial_functions import or2
from magic_torch.horizontal_data import dLh
from magic_torch.precision import CDTYPE
from magic_torch.params import n_r_max, l_chemical_conv

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# === Capture dxidt.old and dxidt.impl from setup_initial_state ===
np.save(os.path.join(out_dir, "dxidt_old.npy"),
        dt_fields.dxidt.old[:, :, 0].cpu().numpy())
np.save(os.path.join(out_dir, "dxidt_impl.npy"),
        dt_fields.dxidt.impl[:, :, 0].cpu().numpy())

# === Step 1: BPR353 stage 1 captures dxidt.expl and xi_imex_rhs ===
# Manually replicate _one_step_dirk stage 1 to intercept intermediates.
tscheme.dt[0] = 3.0e-4
tscheme.set_weights()
build_all_matrices()
tscheme.istage = 1

# Stage 1: radial_loop computes dxidt.expl
radial_loop()
np.save(os.path.join(out_dir, "dxidt_expl.npy"),
        dt_fields.dxidt.expl[:, :, 0].cpu().numpy())

# Manually compute xi_imex_rhs using tscheme.set_imex_rhs
work = tscheme.set_imex_rhs(dt_fields.dxidt)
np.save(os.path.join(out_dir, "xi_imex_rhs.npy"), work.cpu().numpy())

# Complete step 1 via one_step (which does all 4 DIRK stages properly)
# Reset state first: we already did stage 1 radial_loop manually above,
# but one_step expects a clean start. Reload and redo cleanly.
sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Take full step 1
one_step(1, 3.0e-4)

# === Step 2: grid-space at nR=17 (0-based nR=16) ===
# Post-step-1 fields are now in fields.*. Compute grid-space VXi at nR=16.

# Manually do the inverse SHT + get_nl at nR=16 (Fortran nR=17)
f = fields
N = n_r_max
nR_target = 16  # 0-based index for Fortran's nR=17

# Extract single-radial-level spectral slices: (lm_max, 1) for batched SHT
w_nr = f.w_LMloc[:, nR_target:nR_target+1]
dw_nr = f.dw_LMloc[:, nR_target:nR_target+1]
z_nr = f.z_LMloc[:, nR_target:nR_target+1]
xi_nr = f.xi_LMloc[:, nR_target:nR_target+1]

# Pre-multiply Q inputs by dLh (matching radial_loop's _dLh_2d * w_LMloc)
dLh_c = dLh.to(CDTYPE).unsqueeze(1)  # (lm_max, 1)
Q_vel = dLh_c * w_nr     # radial_loop does: _dLh_2d * f.w_LMloc[:, bulk]
S_vel = dw_nr
T_vel = z_nr

# Inverse SHT for velocity (returns (1, n_theta, n_phi))
vrc, vtc, vpc = torpol_to_spat(Q_vel, S_vel, T_vel)

# Inverse SHT for composition
xic_grid = scal_to_spat(xi_nr)

# Compute VXi manually (avoiding get_nl's batched _or2_3 broadcasting issue)
or2_val = or2[nR_target]  # scalar
VXir = vrc * xic_grid
VXit = or2_val * vtc * xic_grid
VXip = or2_val * vpc * xic_grid

# Grid-space outputs have shape (1, n_theta, n_phi) — squeeze batch dim
np.save(os.path.join(out_dir, "VXir_nR17.npy"), VXir[0].cpu().numpy())
np.save(os.path.join(out_dir, "VXit_nR17.npy"), VXit[0].cpu().numpy())
np.save(os.path.join(out_dir, "VXip_nR17.npy"), VXip[0].cpu().numpy())

# Forward SHT to get spectral VXi at nR17
from magic_torch.sht import spat_to_sphertor, scal_to_SH

# Spectral outputs have shape (lm_max, 1) — squeeze batch dim
# No per-radial-level truncation (l_R) in Python — uses full l_max
dVXirLM_nr = scal_to_SH(VXir)
VXitLM_nr, VXipLM_nr = spat_to_sphertor(VXit, VXip)
np.save(os.path.join(out_dir, "dVXirLM_nR17.npy"), dVXirLM_nr[:, 0].cpu().numpy())
np.save(os.path.join(out_dir, "VXitLM_nR17.npy"), VXitLM_nr[:, 0].cpu().numpy())
np.save(os.path.join(out_dir, "VXipLM_nR17.npy"), VXipLM_nr[:, 0].cpu().numpy())

# Also capture dwdt/dzdt intermediates (from step 1 — already stored by setup + radial_loop)
# These were overwritten by the full step 1. Re-read them from the Fortran comparison:
# dwdt, dzdt refs are from current_time_step==1 which fires in the Fortran LM loop,
# meaning they are from the LAST stage of step 1 (stage 4 for DIRK).
# We cannot intercept these mid-step without major refactoring. So we compare
# only the per-step-end field values (already covered by test_dd_multistep).
# Skip dwdt/dzdt intermediates here — they require different dump timing.

print("DD nonlinear runner completed")
