"""IC grid runner: set env vars, import, dump IC grid arrays."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"

import sys
import numpy as np

from magic_torch.radial_functions import (
    r_ic, O_r_ic, cheb_ic, dcheb_ic, d2cheb_ic,
    dr_top_ic, cheb_norm_ic, dr_fac_ic,
)

out_dir = sys.argv[1]
np.save(os.path.join(out_dir, "r_ic.npy"), r_ic.cpu().numpy())
np.save(os.path.join(out_dir, "O_r_ic.npy"), O_r_ic.cpu().numpy())
np.save(os.path.join(out_dir, "cheb_ic.npy"), cheb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dcheb_ic.npy"), dcheb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "d2cheb_ic.npy"), d2cheb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dr_top_ic.npy"), dr_top_ic.cpu().numpy())
np.save(os.path.join(out_dir, "cheb_norm_ic.npy"), np.array(cheb_norm_ic))
np.save(os.path.join(out_dir, "dr_fac_ic.npy"), np.array(dr_fac_ic))

print("IC grid runner completed")
