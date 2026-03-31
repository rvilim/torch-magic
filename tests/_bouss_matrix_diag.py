"""Diagnostic: compare Python vs Fortran coupled bMat for boussBenchSat.

Dumps the coupled bMat and jMat for a specific l value,
and also runs one step to capture the RHS and solution vectors.
"""
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
from magic_torch.step_time import setup_initial_state, initialize_dt, radial_loop, lm_loop
from magic_torch import fields, dt_fields
from magic_torch.params import n_r_max, n_r_ic_max, n_r_tot, n_cheb_ic_max, lm_max, l_max
from magic_torch.time_scheme import tscheme
from magic_torch.blocking import st_lm2, st_lm2l

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(2.0e-4)

print(f"n_r_max={n_r_max}, n_r_ic_max={n_r_ic_max}, n_r_tot={n_r_tot}")
print(f"n_cheb_ic_max={n_cheb_ic_max}, l_max={l_max}, lm_max={lm_max}")

# Dump initial IC old/impl terms for dbdt_ic and djdt_ic
for name in ('dbdt_ic', 'djdt_ic'):
    ta = getattr(dt_fields, name)
    np.save(os.path.join(out_dir, f'{name}_old_init.npy'), ta.old.cpu().numpy())
    np.save(os.path.join(out_dir, f'{name}_impl_init.npy'), ta.impl.cpu().numpy())
    np.save(os.path.join(out_dir, f'{name}_expl_init.npy'), ta.expl.cpu().numpy())

# Also dump initial OC old/impl for dbdt/djdt
for name in ('dbdt', 'djdt', 'dzdt', 'dwdt', 'dsdt', 'dpdt'):
    ta = getattr(dt_fields, name)
    np.save(os.path.join(out_dir, f'{name}_old_init.npy'), ta.old.cpu().numpy())
    np.save(os.path.join(out_dir, f'{name}_impl_init.npy'), ta.impl.cpu().numpy())

# Now do stage 1 of DIRK
tscheme.dt[0] = 2.0e-4
tscheme.set_weights()

from magic_torch.step_time import build_all_matrices
build_all_matrices()

tscheme.istage = 1

# Stage 1: radial_loop
radial_loop()

# Dump explicit terms after radial_loop
for name in ('dbdt_ic', 'djdt_ic'):
    ta = getattr(dt_fields, name)
    np.save(os.path.join(out_dir, f'{name}_expl_stage1.npy'), ta.expl.cpu().numpy())

for name in ('dbdt', 'djdt', 'dzdt', 'dwdt', 'dsdt', 'dpdt'):
    ta = getattr(dt_fields, name)
    np.save(os.path.join(out_dir, f'{name}_expl_stage1.npy'), ta.expl.cpu().numpy())

# Compute the RHS that would go into the B coupled solve
oc_rhs_b = tscheme.set_imex_rhs(dt_fields.dbdt)
oc_rhs_j = tscheme.set_imex_rhs(dt_fields.djdt)
ic_rhs_b = tscheme.set_imex_rhs(dt_fields.dbdt_ic)
ic_rhs_j = tscheme.set_imex_rhs(dt_fields.djdt_ic)

np.save(os.path.join(out_dir, 'oc_rhs_b_stage1.npy'), oc_rhs_b.cpu().numpy())
np.save(os.path.join(out_dir, 'oc_rhs_j_stage1.npy'), oc_rhs_j.cpu().numpy())
np.save(os.path.join(out_dir, 'ic_rhs_b_stage1.npy'), ic_rhs_b.cpu().numpy())
np.save(os.path.join(out_dir, 'ic_rhs_j_stage1.npy'), ic_rhs_j.cpu().numpy())

# Now do the actual lm_loop (implicit solves)
lm_loop()

# Dump fields after stage 1
for name in ("b_LMloc", "db_LMloc", "ddb_LMloc", "aj_LMloc", "dj_LMloc",
             "b_ic", "db_ic", "ddb_ic", "aj_ic", "dj_ic", "ddj_ic"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f'{name}_stage1.npy'), arr)

np.save(os.path.join(out_dir, 'omega_ic_stage1.npy'), np.array(fields.omega_ic))

# Also dump impl terms after stage 1
for name in ('dbdt_ic', 'djdt_ic'):
    ta = getattr(dt_fields, name)
    np.save(os.path.join(out_dir, f'{name}_impl_stage1.npy'), ta.impl.cpu().numpy())

print("Diagnostic completed successfully")
