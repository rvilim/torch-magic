"""Diagnostic runner 2: dump jMat RHS and solution details."""
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
                                    radial_loop, build_all_matrices)
from magic_torch import fields, dt_fields
from magic_torch.time_scheme import tscheme
from magic_torch.params import n_r_max, n_r_ic_max, n_r_tot, lm_max
from magic_torch.precision import CDTYPE, DEVICE
from magic_torch.update_b import _b_inv_by_l, _j_inv_by_l
from magic_torch.blocking import st_lm2l, st_lm2
from magic_torch.algebra import chunked_solve_complex

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(2.0e-4)

tscheme.dt[0] = 2.0e-4
tscheme.set_weights()
build_all_matrices()
tscheme.istage = 1

# Stage 1 radial_loop
radial_loop()

N = n_r_max
N_ic = n_r_ic_max
NT = n_r_tot

# Manually do what _updateB_coupled does for stage 1
# 1. Assemble RHS
oc_rhs_b = tscheme.set_imex_rhs(dt_fields.dbdt)
oc_rhs_j = tscheme.set_imex_rhs(dt_fields.djdt)
ic_rhs_b = tscheme.set_imex_rhs(dt_fields.dbdt_ic)
ic_rhs_j = tscheme.set_imex_rhs(dt_fields.djdt_ic)

np.save(os.path.join(out_dir, "oc_rhs_b.npy"), oc_rhs_b.cpu().numpy())
np.save(os.path.join(out_dir, "oc_rhs_j.npy"), oc_rhs_j.cpu().numpy())
np.save(os.path.join(out_dir, "ic_rhs_b.npy"), ic_rhs_b.cpu().numpy())
np.save(os.path.join(out_dir, "ic_rhs_j.npy"), ic_rhs_j.cpu().numpy())

# Build coupled RHS
rhs_b = torch.zeros(lm_max, NT, dtype=CDTYPE, device=DEVICE)
rhs_b[:, 1:N-1] = oc_rhs_b[:, 1:N-1]
rhs_b[:, N+1:NT] = ic_rhs_b[:, 1:N_ic]

rhs_j = torch.zeros(lm_max, NT, dtype=CDTYPE, device=DEVICE)
rhs_j[:, 1:N-1] = oc_rhs_j[:, 1:N-1]
rhs_j[:, N+1:NT] = ic_rhs_j[:, 1:N_ic]

np.save(os.path.join(out_dir, "coupled_rhs_b.npy"), rhs_b.cpu().numpy())
np.save(os.path.join(out_dir, "coupled_rhs_j.npy"), rhs_j.cpu().numpy())

# Compare at specific lm modes
# l=1, m=0
lm10 = st_lm2[1, 0].item()
# l=4, m=4 
lm44 = st_lm2[4, 4].item()

print(f"lm(1,0)={lm10}, lm(4,4)={lm44}")
print(f"\n=== RHS at l=1,m=0 (lm={lm10}) ===")
print(f"OC rhs_b[{lm10},:5] = {oc_rhs_b[lm10, :5].cpu().numpy()}")
print(f"OC rhs_j[{lm10},:5] = {oc_rhs_j[lm10, :5].cpu().numpy()}")
print(f"IC rhs_b[{lm10},:5] = {ic_rhs_b[lm10, :5].cpu().numpy()}")
print(f"IC rhs_j[{lm10},:5] = {ic_rhs_j[lm10, :5].cpu().numpy()}")

print(f"\n=== dbdt_ic components at l=1,m=0 ===")
print(f"dbdt_ic.old[{lm10},:5,0] = {dt_fields.dbdt_ic.old[lm10, :5, 0].cpu().numpy()}")
print(f"dbdt_ic.impl[{lm10},:5,0] = {dt_fields.dbdt_ic.impl[lm10, :5, 0].cpu().numpy()}")
print(f"dbdt_ic.expl[{lm10},:5,0] = {dt_fields.dbdt_ic.expl[lm10, :5, 0].cpu().numpy()}")

print(f"\n=== djdt_ic components at l=1,m=0 ===")
print(f"djdt_ic.old[{lm10},:5,0] = {dt_fields.djdt_ic.old[lm10, :5, 0].cpu().numpy()}")
print(f"djdt_ic.impl[{lm10},:5,0] = {dt_fields.djdt_ic.impl[lm10, :5, 0].cpu().numpy()}")
print(f"djdt_ic.expl[{lm10},:5,0] = {dt_fields.djdt_ic.expl[lm10, :5, 0].cpu().numpy()}")

# Solve and show results for l=1,m=0
b_sol = chunked_solve_complex(_b_inv_by_l, st_lm2l, rhs_b)
j_sol = chunked_solve_complex(_j_inv_by_l, st_lm2l, rhs_j)

print(f"\n=== Solution at l=1,m=0 ===")
print(f"b_sol[{lm10},:5] (OC cheb) = {b_sol[lm10, :5].cpu().numpy()}")
print(f"b_sol[{lm10},N:N+5] (IC cheb) = {b_sol[lm10, N:N+5].cpu().numpy()}")
print(f"j_sol[{lm10},:5] (OC cheb) = {j_sol[lm10, :5].cpu().numpy()}")
print(f"j_sol[{lm10},N:N+5] (IC cheb) = {j_sol[lm10, N:N+5].cpu().numpy()}")

# Check matrix difference between bMat and jMat for l=1
print(f"\n=== Matrix diff bMat vs jMat for l=1 ===")
b_inv_l1 = _b_inv_by_l[1].cpu()
j_inv_l1 = _j_inv_by_l[1].cpu()
diff = (b_inv_l1 - j_inv_l1).abs().max()
print(f"max |b_inv - j_inv| for l=1: {diff:.6e}")

# Check the actual RHS going into the solvers at l=1,m=0
print(f"\n=== Coupled RHS at l=1,m=0 ===")
print(f"rhs_b[{lm10},:] = ", rhs_b[lm10].cpu().numpy())
print(f"rhs_j[{lm10},:] = ", rhs_j[lm10].cpu().numpy())

print("\nDiagnostic 2 completed")
