#!/usr/bin/env python3
"""Profile the fused lm_loop on MPS."""
import os
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_DEVICE"] = "mps"
os.environ["MAGIC_RA"] = "4.8e4"
os.environ["MAGIC_RAXI"] = "1.2e5"
os.environ["MAGIC_EK"] = "1.0e-3"
os.environ["MAGIC_PR"] = "0.3"
os.environ["MAGIC_SC"] = "3.0"
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_MODE"] = "1"
os.environ["MAGIC_START_FILE"] = "../samples/doubleDiffusion/checkpoint_end.start"
os.environ["MAGIC_DTMAX"] = "3.0e-4"

import magic_torch.step_time as st
st._profile_lm = True

from magic_torch.main import run
# Run 100 steps, then print profile
cfg = {"n_steps": 100, "log_every": 100, "checkpoint_every": 0,
       "output_dir": "./output/profile_lm"}
run(cfg)

# Print results
import numpy as np
print("\n=== Fused lm_loop profiling (100 steps, 4 stages each) ===")
for key in sorted(st._profile_accum.keys()):
    vals = st._profile_accum[key]
    arr = np.array(vals[4:]) * 1000  # skip warmup, convert to ms
    print(f"  {key}: mean={arr.mean():.3f}ms  std={arr.std():.3f}ms  total={arr.sum():.1f}ms")

total = sum(np.array(v[4:]).mean() * 1000 for v in st._profile_accum.values())
print(f"\n  TOTAL per lm_loop call: {total:.2f}ms")
print(f"  Per step (4 calls): {total * 4:.2f}ms")
