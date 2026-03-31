"""Anelastic nonlinear term runner: set env vars, run radial_loop_anel, dump results.

Dumps dt_field components (old/impl/expl for dwdt/dzdt) and IMEX RHS (w/z/p)
after one radial_loop_anel call on the initial state.
"""
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

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import (
    setup_initial_state, initialize_dt, build_all_matrices, radial_loop_anel,
)
from magic_torch.time_scheme import tscheme
from magic_torch.params import dtmax
from magic_torch import dt_fields

out_dir = sys.argv[1]

# Initialize
initialize_fields()
setup_initial_state()
initialize_dt(dtmax)

# Set up time scheme weights (same as step 1 of one_step_cnab2)
tscheme.dt[0] = dtmax
tscheme.dt[1] = dtmax
tscheme.set_weights()
build_all_matrices()

# Run radial loop (populates dt_fields)
radial_loop_anel()

# Dump dt_field components
def save(name, tensor):
    arr = tensor.cpu().to(torch.complex128).numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

# dt_field old/impl/expl for dwdt and dzdt
save("dwdt_old", dt_fields.dwdt.old[:, :, 0])
save("dwdt_impl", dt_fields.dwdt.impl[:, :, 0])
save("dwdt_expl", dt_fields.dwdt.expl[:, :, 0])
save("dzdt_old", dt_fields.dzdt.old[:, :, 0])
save("dzdt_impl", dt_fields.dzdt.impl[:, :, 0])
save("dzdt_expl", dt_fields.dzdt.expl[:, :, 0])

# IMEX RHS assembly
save("w_imex_rhs", tscheme.set_imex_rhs(dt_fields.dwdt))
save("z_imex_rhs", tscheme.set_imex_rhs(dt_fields.dzdt))
save("p_imex_rhs", tscheme.set_imex_rhs(dt_fields.dpdt))

print("Anelastic nonlinear runner completed")
