"""Anelastic init runner: set env vars, initialize fields, dump results."""
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

import sys
import numpy as np
import torch

from magic_torch.init_fields import initialize_fields, _ps_cond_anel
from magic_torch.cosine_transform import costf
from magic_torch import fields

# Get float64 conduction state before any casting
s0_f64, p0_f64 = _ps_cond_anel()
# _ps_cond_anel returns DTYPE/DEVICE after costf+cast, but we need float64.
# Actually let me re-run the solver to get float64 output.
# Since DEVICE=cpu and DTYPE=float64, the return should already be float64.

# Initialize all fields
initialize_fields()

# Save results
out_dir = sys.argv[1]

# Save conduction state (lm=0 mode) in float64
np.save(os.path.join(out_dir, "s0_cond.npy"),
        fields.s_LMloc[0, :].cpu().to(torch.complex128).numpy())
np.save(os.path.join(out_dir, "p0_cond.npy"),
        fields.p_LMloc[0, :].cpu().to(torch.complex128).numpy())

# Save all init fields
field_map = {
    "s_init": fields.s_LMloc,
    "p_init": fields.p_LMloc,
    "w_init": fields.w_LMloc,
    "z_init": fields.z_LMloc,
    "dw_init": fields.dw_LMloc,
    "dz_init": fields.dz_LMloc,
}

for name, tensor in field_map.items():
    arr = tensor.cpu().to(torch.complex128).numpy()
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)

print("Anelastic init runner completed successfully")
