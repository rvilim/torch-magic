"""IC init runner: set env vars, import, run init, dump IC fields."""
import os
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"

import sys
import numpy as np

from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state
from magic_torch import fields

initialize_fields()
setup_initial_state()

out_dir = sys.argv[1]

# IC fields after init + get_mag_ic_rhs_imp
np.save(os.path.join(out_dir, "b_ic_init.npy"), fields.b_ic.cpu().numpy())
np.save(os.path.join(out_dir, "db_ic_init.npy"), fields.db_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddb_ic_init.npy"), fields.ddb_ic.cpu().numpy())
np.save(os.path.join(out_dir, "aj_ic_init.npy"), fields.aj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "dj_ic_init.npy"), fields.dj_ic.cpu().numpy())
np.save(os.path.join(out_dir, "ddj_ic_init.npy"), fields.ddj_ic.cpu().numpy())

# OC fields (different from dynamo_benchmark when l_cond_ic)
np.save(os.path.join(out_dir, "b_init.npy"), fields.b_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "db_init.npy"), fields.db_LMloc.cpu().numpy())
np.save(os.path.join(out_dir, "aj_init.npy"), fields.aj_LMloc.cpu().numpy())

print("IC init runner completed")
