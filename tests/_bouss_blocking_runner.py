"""Blocking runner for boussBenchSat config."""
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
from magic_torch.blocking import st_lm2l, st_lm2m, st_lm2lmA, st_lm2lmS
from magic_torch.params import lm_max, l_max, m_max, minc

out_dir = sys.argv[1]
np.save(os.path.join(out_dir, "lm2l.npy"), st_lm2l.cpu().numpy())
np.save(os.path.join(out_dir, "lm2m.npy"), st_lm2m.cpu().numpy())
np.save(os.path.join(out_dir, "lm2lmA.npy"), st_lm2lmA.cpu().numpy())
np.save(os.path.join(out_dir, "lm2lmS.npy"), st_lm2lmS.cpu().numpy())
np.save(os.path.join(out_dir, "lm_max.npy"), np.array(lm_max))
np.save(os.path.join(out_dir, "l_max.npy"), np.array(l_max))
np.save(os.path.join(out_dir, "minc.npy"), np.array(minc))
print("Blocking runner completed")
