#!/usr/bin/env python3
"""Run FD benchmark profiler on Modal GPU.

Usage:
    uv run modal run benchmark_modal.py --lmax 64 --steps 5
    MAGIC_MODAL_GPU=A100 uv run modal run benchmark_modal.py --lmax 128 --steps 3
"""
import os
import modal

GPU_TYPE = os.environ.get("MAGIC_MODAL_GPU", "H100")

app = modal.App("magic-torch-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pyyaml")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("benchmark_fd_profile.py", remote_path="/root/benchmark_fd_profile.py")
)


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def run_benchmark(lmax: int, steps: int):
    import sys
    sys.path.insert(0, "/root/src")
    os.environ["MAGIC_DEVICE"] = "cuda"
    os.environ["MAGIC_RADIAL_SCHEME"] = "FD"
    os.environ["MAGIC_NR"] = str(2 * lmax + 1)
    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_FD_ORDER"] = "2"
    os.environ["MAGIC_FD_ORDER_BOUND"] = "2"
    os.environ["MAGIC_FD_STRETCH"] = "0.3"
    os.environ["MAGIC_FD_RATIO"] = "0.1"
    os.environ["MAGIC_INIT_S1"] = "404"
    os.environ["MAGIC_AMP_S1"] = "0.1"

    import torch
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    # Run the profiler
    import time
    from collections import defaultdict

    from magic_torch.precision import DEVICE
    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import (
        setup_initial_state, initialize_dt, one_step,
    )
    from magic_torch import step_time, update_s, update_z, update_b
    try:
        from magic_torch import update_w_doublecurl
    except ImportError:
        update_w_doublecurl = None

    _timings = defaultdict(float)
    _counts = defaultdict(int)

    def _sync():
        if DEVICE.type == "cuda":
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

    def _wrap(mod, funcname, label):
        orig = getattr(mod, funcname)
        def timed(*a, **kw):
            with Timer(label):
                return orig(*a, **kw)
        setattr(mod, funcname, timed)

    _wrap(update_s, "updateS", "updateS")
    _wrap(update_z, "updateZ", "updateZ")
    _wrap(update_b, "updateB", "updateB")
    if update_w_doublecurl:
        _wrap(update_w_doublecurl, "updateW", "updateW")

    _orig_rl = step_time.radial_loop
    _orig_lm = step_time.lm_loop
    def _trl():
        with Timer("radial_loop"):
            return _orig_rl()
    def _tlm():
        with Timer("lm_loop"):
            return _orig_lm()
    step_time.radial_loop = _trl
    step_time.lm_loop = _tlm

    from magic_torch import sht as _sht
    for fn in ["scal_to_spat", "torpol_to_spat", "scal_to_SH", "spat_to_sphertor",
               "torpol_to_curl_spat", "pol_to_grad_spat"]:
        if hasattr(_sht, fn):
            _wrap(_sht, fn, f"sht.{fn}")

    from magic_torch import get_nl as _gnl
    _wrap(_gnl, "get_nl", "get_nl")

    from magic_torch import get_td as _gtd
    for fn in ["get_dwdt", "get_dwdt_double_curl", "get_dzdt", "get_dsdt", "get_dbdt"]:
        if hasattr(_gtd, fn):
            _wrap(_gtd, fn, f"td.{fn}")
    _wrap(update_s, "finish_exp_entropy", "finish_exp_s")
    _wrap(update_b, "finish_exp_mag", "finish_exp_mag")
    from magic_torch import courant as _crt
    _wrap(_crt, "courant_check", "courant_check")

    print(f"Profiling FD: device=cuda, l_max={lmax}, N={2*lmax+1}, steps={steps}")
    print()

    initialize_fields()
    setup_initial_state()
    initialize_dt(1e-4)
    one_step(1, 1e-4)
    _timings.clear()
    _counts.clear()

    _sync()
    t0 = time.perf_counter()
    for s in range(2, 2 + steps):
        one_step(s, 1e-4)
    _sync()
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / steps * 1000

    def _avg(label):
        c = _counts.get(label, 0)
        return _timings.get(label, 0) / c if c > 0 else 0.0

    rl = _avg("radial_loop")
    lm = _avg("lm_loop")
    overhead = total_ms - rl - lm

    print(f"=== FD step breakdown (avg of {steps} steps, ms) ===")
    print()
    print(f"radial_loop:        {rl:8.2f} ms  ({rl/total_ms*100:4.1f}%)")
    for label in ["sht.scal_to_spat", "sht.torpol_to_spat", "get_nl",
                  "sht.scal_to_SH", "sht.spat_to_sphertor",
                  "td.get_dwdt_double_curl", "td.get_dwdt", "finish_exp_pol",
                  "td.get_dzdt", "td.get_dsdt", "finish_exp_s",
                  "td.get_dbdt", "finish_exp_mag", "courant_check"]:
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

    return total_ms


@app.local_entrypoint()
def main(lmax: int = 64, steps: int = 5):
    print(f"Launching benchmark on {GPU_TYPE}: l_max={lmax}, steps={steps}")
    result = run_benchmark.remote(lmax, steps)
    print(f"\nResult: {result:.2f} ms/step")
