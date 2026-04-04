"""Runner for testOutputs: full simulation with power/u_square/helicity/hemi output.

Usage: python _testOutputs_runner.py <out_dir>
Env vars must be set before import (see test_testOutputs.py).
"""
import os
import sys

out_dir = sys.argv[1]

os.environ.setdefault("MAGIC_DEVICE", "cpu")

from magic_torch.main import run

cfg = {
    "n_steps": 100,
    "dt": float(os.environ.get("MAGIC_DTMAX", "1e-4")),
    "log_every": int(os.environ.get("MAGIC_N_LOG_STEP", "10")),
    "output_dir": out_dir,
    "tag": "start",
    "n_graphs": 0,
    "checkpoint_every": 0,
}

run(cfg)
print("testOutputs runner done")
