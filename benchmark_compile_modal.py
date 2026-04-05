#!/usr/bin/env python3
"""Benchmark torch.compile on Modal GPU.

Tests multiple compile strategies and compares to eager baseline.

Usage:
    uv run modal run benchmark_compile_modal.py --lmax 64 --steps 20
    uv run modal run benchmark_compile_modal.py --lmax 128 --steps 10
"""
import os
import modal

GPU_TYPE = os.environ.get("MAGIC_MODAL_GPU", "H100")

app = modal.App("magic-torch-compile-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pyyaml")
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(image=image, gpu=GPU_TYPE, timeout=1200)
def run_benchmark(lmax: int, steps: int, mode: str, scheme: str):
    """Run benchmark with a given compile mode.

    Args:
        mode: "eager", "compile_one_step", "compile_radial_lm",
              "compile_solvers", "compile_nl_td"
        scheme: "FD" or "cheb"
    """
    import sys
    sys.path.insert(0, "/root/src")
    os.environ["MAGIC_DEVICE"] = "cuda"
    if scheme == "FD":
        os.environ["MAGIC_RADIAL_SCHEME"] = "FD"
        os.environ["MAGIC_FD_ORDER"] = "2"
        os.environ["MAGIC_FD_ORDER_BOUND"] = "2"
        os.environ["MAGIC_FD_STRETCH"] = "0.3"
        os.environ["MAGIC_FD_RATIO"] = "0.1"
    os.environ["MAGIC_NR"] = str(2 * lmax + 1)
    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_MINC"] = "1"
    os.environ["MAGIC_RADRATIO"] = "0.35"
    os.environ["MAGIC_INIT_S1"] = "404"
    os.environ["MAGIC_AMP_S1"] = "0.1"

    import torch
    import time

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {mode}, Scheme: {scheme}, l_max={lmax}, N={2*lmax+1}")
    print()

    from magic_torch.precision import DEVICE
    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import (
        setup_initial_state, initialize_dt, one_step,
        radial_loop, lm_loop,
    )
    from magic_torch import step_time

    # Apply compile based on mode
    compiled_fn = None
    if mode == "compile_one_step":
        # Compile the entire one_step function
        compiled_fn = torch.compile(one_step)

    elif mode == "compile_radial_lm":
        # Compile radial_loop and lm_loop separately
        step_time.radial_loop = torch.compile(radial_loop)
        step_time.lm_loop = torch.compile(lm_loop)

    elif mode == "compile_solvers":
        # Compile individual solver update functions
        from magic_torch import update_s, update_z, update_b
        update_s.updateS = torch.compile(update_s.updateS)
        update_z.updateZ = torch.compile(update_z.updateZ)
        update_b.updateB = torch.compile(update_b.updateB)
        if hasattr(step_time, 'updateS'):
            step_time.updateS = update_s.updateS
        if hasattr(step_time, 'updateZ'):
            step_time.updateZ = update_z.updateZ
        if hasattr(step_time, 'updateB'):
            step_time.updateB = update_b.updateB
        try:
            from magic_torch import update_w_doublecurl
            update_w_doublecurl.updateW = torch.compile(update_w_doublecurl.updateW)
            if hasattr(step_time, 'updateW'):
                step_time.updateW = update_w_doublecurl.updateW
        except ImportError:
            pass

    elif mode == "compile_nl_td":
        # Compile nonlinear + time derivative functions
        from magic_torch import get_nl as _gnl, get_td as _gtd
        _gnl.get_nl = torch.compile(_gnl.get_nl)
        if hasattr(step_time, 'get_nl'):
            step_time.get_nl = _gnl.get_nl
        for fn in ["get_dwdt", "get_dwdt_double_curl", "get_dzdt", "get_dsdt", "get_dbdt"]:
            if hasattr(_gtd, fn):
                setattr(_gtd, fn, torch.compile(getattr(_gtd, fn)))
                if hasattr(step_time, fn):
                    setattr(step_time, fn, getattr(_gtd, fn))

    # Initialize
    initialize_fields()
    setup_initial_state()
    initialize_dt(1e-4)

    step_fn = compiled_fn if compiled_fn else one_step

    # Warmup (includes compilation on first call)
    print("Warmup (includes compile)...")
    torch.cuda.synchronize()
    t_warmup_start = time.perf_counter()
    step_fn(1, 1e-4)
    torch.cuda.synchronize()
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"Warmup: {t_warmup:.1f}s")

    # Second warmup to catch any lazy compilation
    torch.cuda.synchronize()
    t_w2_start = time.perf_counter()
    step_fn(2, 1e-4)
    torch.cuda.synchronize()
    t_w2 = time.perf_counter() - t_w2_start
    print(f"Warmup2: {t_w2*1000:.1f}ms")
    print()

    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(3, 3 + steps):
        step_fn(s, 1e-4)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / steps * 1000

    print(f"Result: {total_ms:.2f} ms/step (avg of {steps} steps)")

    # GPU memory
    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    mem_reserved = torch.cuda.max_memory_reserved() / 1e9
    print(f"GPU memory: {mem_alloc:.1f} GB allocated, {mem_reserved:.1f} GB reserved")

    return total_ms


@app.local_entrypoint()
def main(lmax: int = 64, steps: int = 20, scheme: str = "cheb"):
    modes = ["eager", "compile_nl_td", "compile_solvers",
             "compile_radial_lm", "compile_one_step"]

    print(f"Benchmarking torch.compile on {GPU_TYPE}: l_max={lmax}, "
          f"steps={steps}, scheme={scheme}")
    print("=" * 60)

    results = {}
    for mode in modes:
        print(f"\n--- {mode} ---")
        try:
            ms = run_benchmark.remote(lmax, steps, mode, scheme)
            results[mode] = ms
        except Exception as e:
            print(f"  FAILED: {e}")
            results[mode] = None

    print("\n" + "=" * 60)
    print(f"Summary (l_max={lmax}, N={2*lmax+1}, {scheme}, {GPU_TYPE})")
    print(f"{'Mode':<25s} {'ms/step':>10s} {'vs eager':>10s}")
    print("-" * 45)
    eager_ms = results.get("eager")
    for mode in modes:
        ms = results[mode]
        if ms is None:
            print(f"{mode:<25s} {'FAILED':>10s}")
        else:
            speedup = f"{eager_ms/ms:.2f}x" if eager_ms else "—"
            print(f"{mode:<25s} {ms:>9.2f}ms {speedup:>10s}")
