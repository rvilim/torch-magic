#!/usr/bin/env python
"""Deep per-function profiling for FD time step.

Usage: uv run python benchmark_fd_profile.py [--device cpu|mps] [--lmax 16] [--steps 10]
"""
import os
import sys
import time
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
parser.add_argument("--lmax", type=int, default=16)
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()

nr = 2 * args.lmax + 1
os.environ["MAGIC_DEVICE"] = args.device
os.environ["MAGIC_RADIAL_SCHEME"] = "FD"
os.environ["MAGIC_NR"] = str(nr)
os.environ["MAGIC_LMAX"] = str(args.lmax)
os.environ["MAGIC_FD_ORDER"] = "2"
os.environ["MAGIC_FD_ORDER_BOUND"] = "2"
os.environ["MAGIC_FD_STRETCH"] = "0.3"
os.environ["MAGIC_FD_RATIO"] = "0.1"
os.environ["MAGIC_INIT_S1"] = "404"
os.environ["MAGIC_AMP_S1"] = "0.1"

import torch
from magic_torch.precision import DEVICE
from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
from magic_torch import step_time

# ============================================================
# Timing infrastructure — inject into step_time module
# ============================================================
_timings = defaultdict(float)
_counts = defaultdict(int)


def _sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        _sync()
        dt = (time.perf_counter() - self.t0) * 1000
        _timings[self.label] += dt
        _counts[self.label] += 1


# Inject timer into step_time module so radial_loop/lm_loop can use it
step_time._prof_timer = Timer
step_time._prof_enabled = True

# ============================================================
# Monkey-patch radial_loop to add internal timing
# ============================================================
_orig_radial_loop = step_time.radial_loop


def _profiled_radial_loop():
    """Replacement radial_loop with internal timing."""
    from magic_torch import sht, fields as f, dt_fields as d
    from magic_torch.step_time import (
        _dLh_2d, _or2_bulk_2d, l_mag, l_chemical_conv, l_double_curl,
    )
    from magic_torch.params import n_r_max, lm_max

    T = Timer

    with T("rl.total"):
        # Just call the original — we'll time the top-level sections
        _orig_radial_loop()


# Actually, monkey-patching radial_loop internals is too complex.
# Let's just time the top-level calls: radial_loop, lm_loop, and each update_*.

_orig_one_step = step_time._one_step_cnab2


def _profiled_one_step(n_time_step, dt):
    from magic_torch import time_scheme as ts
    from magic_torch.step_time import (
        radial_loop, radial_loop_anel, lm_loop, build_all_matrices,
        _dtMax,
    )
    from magic_torch.params import l_anel
    from magic_torch.courant import dt_courant

    T = Timer

    with T("total"):
        # 1. Radial loop
        with T("radial_loop"):
            if l_anel:
                radial_loop_anel()
            else:
                radial_loop()

        # 2-5 handled by original
        # Just call original for the rest (dt management + lm_loop)
        # But we need to replicate the logic to time lm_loop separately...
        # This is getting complex. Let me just time the update functions.

    # Fall back to calling original
    pass


# Simpler approach: wrap each update function at module level
def _wrap_update(mod, funcname, label):
    orig = getattr(mod, funcname)
    def timed(*a, **kw):
        with Timer(label):
            return orig(*a, **kw)
    setattr(mod, funcname, timed)
    # Also patch step_time's reference
    if hasattr(step_time, funcname):
        setattr(step_time, funcname, timed)


from magic_torch import update_s, update_z, update_b
try:
    from magic_torch import update_w_doublecurl
except ImportError:
    update_w_doublecurl = None

_wrap_update(update_s, "updateS", "updateS")
_wrap_update(update_z, "updateZ", "updateZ")
_wrap_update(update_b, "updateB", "updateB")
if update_w_doublecurl:
    _wrap_update(update_w_doublecurl, "updateW", "updateW")

# Wrap radial_loop and lm_loop at the step_time module level
_orig_rl = step_time.radial_loop
_orig_lm = step_time.lm_loop


def _timed_rl():
    with Timer("radial_loop"):
        return _orig_rl()


def _timed_lm():
    with Timer("lm_loop"):
        return _orig_lm()


step_time.radial_loop = _timed_rl
step_time.lm_loop = _timed_lm

# Wrap SHT functions — these are called from step_time via module reference
from magic_torch import sht as _sht_mod

for fname in ["scal_to_spat", "torpol_to_spat", "scal_to_SH", "spat_to_sphertor",
              "torpol_to_curl_spat", "pol_to_grad_spat"]:
    if hasattr(_sht_mod, fname):
        _wrap_update(_sht_mod, fname, f"sht.{fname}")

# Wrap get_nl
from magic_torch import get_nl as _gnl_mod
_wrap_update(_gnl_mod, "get_nl", "get_nl")

# Wrap get_td functions
from magic_torch import get_td as _gtd_mod
for fname in ["get_dwdt", "get_dwdt_double_curl", "get_dzdt", "get_dpdt",
              "get_dsdt", "get_dbdt", "get_dxidt"]:
    if hasattr(_gtd_mod, fname):
        _wrap_update(_gtd_mod, fname, f"td.{fname}")

# Wrap finish_exp functions
_wrap_update(update_s, "finish_exp_entropy", "finish_exp_s")
_wrap_update(update_b, "finish_exp_mag", "finish_exp_mag")
try:
    from magic_torch.update_wp import finish_exp_pol
    from magic_torch import update_wp
    _wrap_update(update_wp, "finish_exp_pol", "finish_exp_pol")
except:
    pass

# Wrap courant
from magic_torch import courant as _crt_mod
_wrap_update(_crt_mod, "courant_check", "courant_check")

# ============================================================
# Run
# ============================================================
print(f"Profiling FD: device={args.device}, l_max={args.lmax}, N={nr}, steps={args.steps}")
print()

initialize_fields()
setup_initial_state()
initialize_dt(1e-4)

# Warmup
one_step(1, 1e-4)
_timings.clear()
_counts.clear()

# Timed steps
_sync()
t_total_start = time.perf_counter()
for s in range(2, 2 + args.steps):
    one_step(s, 1e-4)
_sync()
t_total_end = time.perf_counter()

total_ms = (t_total_end - t_total_start) / args.steps * 1000

# ============================================================
# Report
# ============================================================
def _avg(label):
    c = _counts.get(label, 0)
    return _timings.get(label, 0) / c if c > 0 else 0.0


print(f"=== FD step breakdown (avg of {args.steps} steps, ms) ===")
print()

rl = _avg("radial_loop")
lm = _avg("lm_loop")
overhead = total_ms - rl - lm

print(f"radial_loop:        {rl:8.2f} ms  ({rl/total_ms*100:4.1f}%)")
for label in ["sht.scal_to_spat", "sht.torpol_to_spat",
              "sht.torpol_to_curl_spat", "sht.pol_to_grad_spat",
              "get_nl",
              "sht.scal_to_SH", "sht.spat_to_sphertor",
              "td.get_dwdt_double_curl", "td.get_dwdt", "finish_exp_pol",
              "td.get_dzdt", "td.get_dpdt",
              "td.get_dsdt", "finish_exp_s",
              "td.get_dbdt", "finish_exp_mag",
              "courant_check"]:
    v = _avg(label)
    if v > 0.001:
        print(f"  {label:28s} {v:7.3f} ms")

print()
print(f"lm_loop:            {lm:8.2f} ms  ({lm/total_ms*100:4.1f}%)")
for label in ["updateS", "updateZ", "updateW", "updateB"]:
    v = _avg(label)
    if v > 0.001:
        print(f"  {label:28s} {v:7.3f} ms")

print()
print(f"overhead:           {overhead:8.2f} ms  ({overhead/total_ms*100:4.1f}%)")
print(f"total:              {total_ms:8.2f} ms/step")
