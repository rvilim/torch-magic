"""CFL runner for doubleDiffusion (nomag path)."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_RADIAL_CHUNK"] = "0"
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
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt, _radial_loop_dispatch

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

# Load checkpoint (DD starts from restart, not conduction state)
sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Call radial loop once — matches Fortran stage 1 of step 1
_radial_loop_dispatch()
# Get per-level CFL arrays for comparison with Fortran reference
from magic_torch.courant import courant_check
from magic_torch.step_time import _vrc, _vtc, _vpc
from magic_torch.time_scheme import tscheme
dtrkc, dthkc = courant_check(_vrc, _vtc, _vpc,
                              courfac=tscheme.courfac, alffac=tscheme.alffac)
np.save(os.path.join(out_dir, "dtrkc.npy"), dtrkc.cpu().numpy())
np.save(os.path.join(out_dir, "dthkc.npy"), dthkc.cpu().numpy())
print("nomag CFL runner completed")
