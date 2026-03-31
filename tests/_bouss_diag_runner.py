"""Diagnostic runner: dump intermediates during boussBenchSat step 1."""
import os
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

import sys
import numpy as np
import torch

from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import (setup_initial_state, initialize_dt,
                                    radial_loop, lm_loop, build_all_matrices)
from magic_torch import fields, dt_fields
from magic_torch.time_scheme import tscheme

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

# Load checkpoint and setup
sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(2.0e-4)

# Dump initial old/impl for entropy to check setup_initial_state
np.save(os.path.join(out_dir, "dsdt_old_init.npy"), dt_fields.dsdt.old[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dsdt_impl_init.npy"), dt_fields.dsdt.impl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dbdt_old_init.npy"), dt_fields.dbdt.old[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dbdt_impl_init.npy"), dt_fields.dbdt.impl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dbdt_ic_old_init.npy"), dt_fields.dbdt_ic.old[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dbdt_ic_impl_init.npy"), dt_fields.dbdt_ic.impl[:,:,0].cpu().numpy())

# Start DIRK step: set dt, weights, build matrices
tscheme.dt[0] = 2.0e-4
tscheme.set_weights()
build_all_matrices()
tscheme.istage = 1

# === Stage 1: radial_loop (explicit) ===
radial_loop()

# Dump explicit terms after stage 1 radial_loop
np.save(os.path.join(out_dir, "dsdt_expl_s1.npy"), dt_fields.dsdt.expl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dwdt_expl_s1.npy"), dt_fields.dwdt.expl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dzdt_expl_s1.npy"), dt_fields.dzdt.expl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dbdt_expl_s1.npy"), dt_fields.dbdt.expl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "djdt_expl_s1.npy"), dt_fields.djdt.expl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "dbdt_ic_expl_s1.npy"), dt_fields.dbdt_ic.expl[:,:,0].cpu().numpy())
np.save(os.path.join(out_dir, "djdt_ic_expl_s1.npy"), dt_fields.djdt_ic.expl[:,:,0].cpu().numpy())

# Dump IMEX RHS for s before solve (what goes into the solver)
rhs_s = tscheme.set_imex_rhs(dt_fields.dsdt)
np.save(os.path.join(out_dir, "s_imex_rhs_s1.npy"), rhs_s.cpu().numpy())
rhs_b = tscheme.set_imex_rhs(dt_fields.dbdt)
np.save(os.path.join(out_dir, "b_imex_rhs_s1.npy"), rhs_b.cpu().numpy())

# === Stage 1: lm_loop (implicit solve) ===
lm_loop()

# Dump fields after stage 1 solve
for name in ("w_LMloc", "dw_LMloc", "z_LMloc", "dz_LMloc",
             "s_LMloc", "ds_LMloc", "p_LMloc", "dp_LMloc",
             "b_LMloc", "db_LMloc", "aj_LMloc", "dj_LMloc",
             "b_ic", "db_ic", "aj_ic", "dj_ic"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}_after_s1.npy"), arr)
np.save(os.path.join(out_dir, "omega_ic_after_s1.npy"), np.array(fields.omega_ic))

# Continue stages 2-4
for n_stage in range(2, tscheme.nstages + 1):
    tscheme.istage = n_stage
    if tscheme.l_exp_calc[n_stage - 1]:
        radial_loop()
    lm_loop()
    # Dump after each stage
    for name in ("s_LMloc", "b_LMloc", "z_LMloc", "aj_LMloc", "b_ic", "aj_ic"):
        arr = getattr(fields, name).cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}_after_s{n_stage}.npy"), arr)
    np.save(os.path.join(out_dir, f"omega_ic_after_s{n_stage}.npy"), np.array(fields.omega_ic))

print("Diagnostic runner completed")
