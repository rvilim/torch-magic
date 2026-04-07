#!/usr/bin/env python3
"""Benchmark: SHTns GPU vs our batched bmm SHT.

Measures single-transform GPU timing to decide if SHTns integration is worth pursuing.

Usage:
    MAGIC_MODAL_GPU=B200 uv run modal run scripts/bench_sht_shtns.py --lmax 384
    uv run modal run scripts/bench_sht_shtns.py --lmax 128
"""

import os
import modal

GPU_TYPE = os.environ.get("MAGIC_MODAL_GPU", "H100")

app = modal.App("sht-shtns-bench")

SHTNS_SRC = "/Users/rvilim/dynamo/master"

image = (
    # Use CUDA base image so nvcc and CUDA libs are available at build time
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libfftw3-dev", "build-essential", "autoconf", "libtool")
    .pip_install("torch", "numpy")
    # Upload full SHTns source (includes GPU kernel generators)
    .add_local_dir(SHTNS_SRC, remote_path="/root/shtns_src", copy=True)
    .run_commands(
        "cd /root/shtns_src && CUDA_PATH=/usr/local/cuda pip install .",
    )
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def run_benchmark(lmax: int, n_batch: int):
    import sys
    import time
    import numpy as np

    print(f"GPU: {__import__('torch').cuda.get_device_name()}")

    # --- SHTns setup ---
    import shtns
    print(f"SHTns version: {shtns.__version__ if hasattr(shtns, '__version__') else 'unknown'}")

    sh = shtns.sht(lmax, lmax, mres=1, norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE)

    # Try to set batching via ctypes
    batched = False
    try:
        import ctypes
        import _shtns_cuda as _shtns_mod
        lib_path = _shtns_mod.__file__
        lib = ctypes.CDLL(lib_path)
        lib.shtns_set_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        lib.shtns_set_many.restype = ctypes.c_int
        # Extract the internal shtns_cfg pointer from SWIG object
        cfg_ptr = ctypes.c_void_p(int(sh.this))
        ret = lib.shtns_set_many(cfg_ptr, n_batch, sh.nlm)
        if ret > 0:
            print(f"shtns_set_many({n_batch}, {sh.nlm}) = {ret} — batched mode OK")
            batched = True
        else:
            print(f"shtns_set_many failed (ret={ret}), using single-transform mode")
    except Exception as e:
        print(f"Cannot set batching: {e}")

    # Set grid with GPU
    try:
        sh.set_grid(flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)
        print(f"Grid: nlat={sh.nlat}, nphi={sh.nphi}, nlm={sh.nlm}, spat_shape={sh.spat_shape}")
        sh.print_info()
        gpu_ok = True
    except Exception as e:
        print(f"GPU grid setup failed: {e}, falling back to CPU")
        sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
        gpu_ok = False

    # --- Our bmm SHT setup ---
    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_DEVICE"] = "cuda"
    sys.path.insert(0, "/root/src")
    t0 = time.perf_counter()
    from magic_torch.sht import scal_to_spat, torpol_to_spat
    from magic_torch.params import lm_max, n_m_max, n_theta_max, n_phi_max
    from magic_torch.precision import CDTYPE, DTYPE, DEVICE
    import torch
    t_import = time.perf_counter() - t0
    print(f"\nmagic_torch import: {t_import:.1f}s")
    print(f"lm_max={lm_max}, n_theta={n_theta_max}, n_phi={n_phi_max}")

    # --- Create test data ---
    torch.manual_seed(42)
    np.random.seed(42)

    # For our bmm: (lm_max, n_batch) complex128 on CUDA
    Slm_torch = (torch.randn(lm_max, n_batch, dtype=DTYPE, device=DEVICE)
                 + 1j * torch.randn(lm_max, n_batch, dtype=DTYPE, device=DEVICE)).to(CDTYPE)

    # For SHTns: need proper layout
    if batched:
        # Batched: (n_batch, nlm) complex128 contiguous — batch outermost, lm innermost
        Slm_np = Slm_torch.T.contiguous().cpu().numpy()  # (n_batch, lm_max) C-contiguous
    else:
        # Single: (nlm,) complex128
        Slm_np = Slm_torch[:, 0].cpu().numpy()

    # --- Benchmark our bmm ---
    print(f"\n--- scal_to_spat benchmark (n_batch={n_batch}) ---")

    # Warmup
    for _ in range(5):
        _ = scal_to_spat(Slm_torch)
    torch.cuda.synchronize()

    # Time bmm
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_iter = 50
    start.record()
    for _ in range(n_iter):
        _ = scal_to_spat(Slm_torch)
    end.record()
    torch.cuda.synchronize()
    bmm_ms = start.elapsed_time(end) / n_iter
    print(f"  bmm: {bmm_ms:.3f} ms/call")

    # --- Benchmark SHTns ---
    if gpu_ok:
        try:
            import cupy
            has_cupy = True
        except ImportError:
            has_cupy = False
            print("  (no cupy — using raw pointer API)")

        if batched:
            # Batched: allocate GPU arrays via torch (CuPy interop via data_ptr)
            slm_gpu = torch.tensor(Slm_np, dtype=CDTYPE, device=DEVICE).contiguous()
            spat_shape_batched = (n_batch, sh.nphi, sh.nlat)
            out_gpu = torch.zeros(spat_shape_batched, dtype=DTYPE, device=DEVICE).contiguous()

            # Warmup
            torch.cuda.synchronize()
            for _ in range(5):
                sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
            torch.cuda.synchronize()

            # Time
            start.record()
            for _ in range(n_iter):
                sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
            end.record()
            torch.cuda.synchronize()
            shtns_ms = start.elapsed_time(end) / n_iter
            print(f"  shtns (batched={n_batch}): {shtns_ms:.3f} ms/call")
        else:
            # Single transform: loop over n_batch
            slm_gpu = torch.tensor(Slm_np, dtype=CDTYPE, device=DEVICE).contiguous()
            out_gpu = torch.zeros(sh.spat_shape, dtype=DTYPE, device=DEVICE).contiguous()

            # Warmup
            torch.cuda.synchronize()
            for _ in range(3):
                sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
            torch.cuda.synchronize()

            # Time single transform
            start.record()
            for _ in range(n_iter):
                sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
            end.record()
            torch.cuda.synchronize()
            shtns_single_ms = start.elapsed_time(end) / n_iter
            print(f"  shtns (single): {shtns_single_ms:.3f} ms/call")

            # Extrapolate n_batch serial calls
            shtns_ms = shtns_single_ms * n_batch
            print(f"  shtns ({n_batch} serial calls): {shtns_ms:.3f} ms (extrapolated)")

        print(f"\n  Speedup: {bmm_ms / shtns_ms:.2f}x")
    else:
        print("  SHTns GPU not available, skipping")

    print("\nDone!")


@app.local_entrypoint()
def main(lmax: int = 128, n_batch: int = 32):
    print(f"Launching SHTns benchmark: l_max={lmax}, n_batch={n_batch}, GPU={GPU_TYPE}")
    run_benchmark.remote(lmax, n_batch)
