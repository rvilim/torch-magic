"""DD init runner: load checkpoint, dump initial fields."""
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
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt
from magic_torch import fields
from magic_torch.params import l_mag

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]

sim_time = load_fortran_checkpoint(ckpt_path)
setup_initial_state()
initialize_dt(3.0e-4)

# Dump xi/dxi init fields (dxi computed by setup_initial_state via get_ddr)
for name in ("xi_LMloc", "dxi_LMloc", "b_LMloc", "aj_LMloc",
             "p_LMloc", "s_LMloc", "w_LMloc"):
    arr = getattr(fields, name).cpu().numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

# Dump l_mag flag
np.save(os.path.join(out_dir, "l_mag.npy"), np.array(l_mag))

print(f"DD init runner completed, sim_time={sim_time}")
