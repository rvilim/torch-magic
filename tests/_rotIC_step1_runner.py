"""RotIC step1 runner: set env vars, import, run 1 step, dump fields."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"
os.environ["MAGIC_NROTIC"] = "1"

import sys
import numpy as np

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.params import dtmax
from magic_torch import fields

initialize_fields()
setup_initial_state()
initialize_dt(dtmax)
one_step(1, dtmax)

out_dir = sys.argv[1]

# IC fields after step 1
np.save(os.path.join(out_dir, "b_ic_step1.npy"), fields.b_ic.cpu().numpy())
np.save(os.path.join(out_dir, "db_ic_step1.npy"), fields.db_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_ic_step1.npy"), fields.ddb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "aj_ic_step1.npy"), fields.aj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dj_ic_step1.npy"), fields.dj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddj_ic_step1.npy"), fields.ddj_ic.cpu().numpy())

# OC fields after step 1
np.save(os.path.join(out_dir, "b_step1.npy"), fields.b_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "db_step1.npy"), fields.db_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_step1.npy"), fields.ddb_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "aj_step1.npy"), fields.aj_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "dj_step1.npy"), fields.dj_LMloc.cpu().numpy())

# Non-magnetic OC fields
np.save(os.path.join(out_dir, "w_step1.npy"), fields.w_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "z_step1.npy"), fields.z_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "s_step1.npy"), fields.s_LMloc.cpu().numpy())

# IC rotation rate
np.save(os.path.join(out_dir, "omega_ic_step1.npy"), np.array(fields.omega_ic))

print("RotIC step1 runner completed")
