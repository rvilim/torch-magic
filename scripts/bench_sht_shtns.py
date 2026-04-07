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
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libfftw3-dev", "build-essential", "autoconf", "libtool")
    .pip_install("torch", "numpy")
    # Upload full SHTns source (includes GPU kernel generators)
    .add_local_dir(SHTNS_SRC, remote_path="/root/shtns_src", copy=True)
    .run_commands(
        # Force gcc as the C compiler (NVIDIA image may default to clang)
        "cd /root/shtns_src && CC=gcc CUDA_PATH=/usr/local/cuda pip install .",
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

    # Get our grid sizes for matching
    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_DEVICE"] = "cuda"
    sys.path.insert(0, "/root/src")
    t0 = time.perf_counter()
    from magic_torch.params import lm_max, n_r_max, n_m_max, n_theta_max, n_phi_max
    from magic_torch.precision import CDTYPE, DTYPE, DEVICE
    import torch
    t_import = time.perf_counter() - t0
    print(f"magic_torch import: {t_import:.1f}s")
    print(f"lm_max={lm_max}, n_theta={n_theta_max}, n_phi={n_phi_max}")

    # Try to set batching via ctypes
    batched = False
    try:
        import ctypes
        import _shtns_cuda as _shtns_mod
        lib_path = _shtns_mod.__file__
        lib = ctypes.CDLL(lib_path)
        lib.shtns_set_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        lib.shtns_set_many.restype = ctypes.c_int
        cfg_ptr = ctypes.c_void_p(int(sh.this))
        ret = lib.shtns_set_many(cfg_ptr, n_batch, sh.nlm)
        if ret > 0:
            print(f"shtns_set_many({n_batch}, {sh.nlm}) = {ret} — batched mode OK")
            batched = True
        else:
            print(f"shtns_set_many failed (ret={ret}), using single-transform mode")
    except Exception as e:
        print(f"Cannot set batching: {e}")

    # Set grid with GPU — use OUR grid sizes to match
    try:
        sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                     flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)
        print(f"Grid: nlat={sh.nlat}, nphi={sh.nphi}, nlm={sh.nlm}, spat_shape={sh.spat_shape}")
        sh.print_info()
        gpu_ok = True
    except Exception as e:
        print(f"GPU grid setup failed: {e}, falling back to CPU")
        sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                     flags=shtns.SHT_THETA_CONTIGUOUS)
        gpu_ok = False

    # --- Our bmm SHT setup ---
    from magic_torch.sht import scal_to_spat, torpol_to_spat

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


@app.function(image=image, gpu=GPU_TYPE, timeout=1200)
def run_batch_sweep(lmax: int, batch_sizes: list):
    """Test multiple batch sizes in a single container (avoids re-importing magic_torch)."""
    import sys
    import time
    import numpy as np

    print(f"GPU: {__import__('torch').cuda.get_device_name()}")

    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_DEVICE"] = "cuda"
    sys.path.insert(0, "/root/src")
    from magic_torch.params import lm_max, n_theta_max, n_phi_max
    from magic_torch.sht import scal_to_spat, torpol_to_spat
    from magic_torch.precision import CDTYPE, DTYPE, DEVICE
    from magic_torch.horizontal_data import dLh
    import torch
    import shtns
    import ctypes
    import _shtns_cuda as _shtns_mod

    lib = ctypes.CDLL(_shtns_mod.__file__)
    lib.shtns_set_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    lib.shtns_set_many.restype = ctypes.c_int

    _dLh_1d = dLh.to(CDTYPE).unsqueeze(1)  # for torpol pre-multiply

    print(f"lm_max={lm_max}, n_theta={n_theta_max}, n_phi={n_phi_max}\n")

    # === Scalar SHT sweep ===
    print("=== scal_to_spat (scalar forward SHT) ===")
    print(f"{'n_batch':>8} {'bmm ms':>10} {'shtns ms':>10} {'speedup':>8}")
    print("-" * 40)

    for nb in batch_sizes:
        sh = shtns.sht(lmax, lmax, mres=1, norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE)
        lib.shtns_set_many(ctypes.c_void_p(int(sh.this)), nb, sh.nlm)
        sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                     flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)

        torch.manual_seed(42)
        Slm_torch = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
                      + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
        slm_gpu = Slm_torch.T.contiguous()
        out_gpu = torch.zeros(nb, sh.nphi, sh.nlat, dtype=DTYPE, device=DEVICE).contiguous()

        n_iter = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        # BMM
        for _ in range(5): scal_to_spat(Slm_torch)
        torch.cuda.synchronize()
        s.record()
        for _ in range(n_iter): scal_to_spat(Slm_torch)
        e.record()
        torch.cuda.synchronize()
        bmm_ms = s.elapsed_time(e) / n_iter

        # SHTns
        torch.cuda.synchronize()
        for _ in range(5): sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
        torch.cuda.synchronize()
        s.record()
        for _ in range(n_iter): sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
        e.record()
        torch.cuda.synchronize()
        shtns_ms = s.elapsed_time(e) / n_iter

        print(f"{nb:>8} {bmm_ms:>10.3f} {shtns_ms:>10.3f} {bmm_ms/shtns_ms:>7.2f}x")

    # === Vector SHT sweep (torpol_to_spat / cu_SHqst_to_spat) ===
    print(f"\n=== torpol_to_spat (vector forward SHT, the expensive one) ===")
    print(f"{'n_batch':>8} {'bmm ms':>10} {'shtns ms':>10} {'speedup':>8}")
    print("-" * 40)

    for nb in batch_sizes:
        sh = shtns.sht(lmax, lmax, mres=1, norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE)
        lib.shtns_set_many(ctypes.c_void_p(int(sh.this)), nb, sh.nlm)
        sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                     flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)

        torch.manual_seed(42)
        Qlm = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
               + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
        Slm2 = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
                + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
        Tlm = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
               + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)

        # SHTns inputs: (nb, lm_max) contiguous
        q_gpu = Qlm.T.contiguous()
        s_gpu = Slm2.T.contiguous()
        t_gpu = Tlm.T.contiguous()
        vr_gpu = torch.zeros(nb, sh.nphi, sh.nlat, dtype=DTYPE, device=DEVICE).contiguous()
        vt_gpu = torch.zeros_like(vr_gpu)
        vp_gpu = torch.zeros_like(vr_gpu)

        n_iter = 50
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)

        # BMM torpol_to_spat
        for _ in range(3): torpol_to_spat(Qlm, Slm2, Tlm)
        torch.cuda.synchronize()
        s_ev.record()
        for _ in range(n_iter): torpol_to_spat(Qlm, Slm2, Tlm)
        e_ev.record()
        torch.cuda.synchronize()
        bmm_ms = s_ev.elapsed_time(e_ev) / n_iter

        # SHTns cu_SHqst_to_spat
        torch.cuda.synchronize()
        for _ in range(3):
            sh.cu_SHqst_to_spat(q_gpu.data_ptr(), s_gpu.data_ptr(), t_gpu.data_ptr(),
                                 vr_gpu.data_ptr(), vt_gpu.data_ptr(), vp_gpu.data_ptr())
        torch.cuda.synchronize()
        s_ev.record()
        for _ in range(n_iter):
            sh.cu_SHqst_to_spat(q_gpu.data_ptr(), s_gpu.data_ptr(), t_gpu.data_ptr(),
                                 vr_gpu.data_ptr(), vt_gpu.data_ptr(), vp_gpu.data_ptr())
        e_ev.record()
        torch.cuda.synchronize()
        shtns_ms = s_ev.elapsed_time(e_ev) / n_iter

        print(f"{nb:>8} {bmm_ms:>10.3f} {shtns_ms:>10.3f} {bmm_ms/shtns_ms:>7.2f}x")


@app.function(image=image, gpu=GPU_TYPE, timeout=3600)
def run_full_sweep(lmax_list: list, batch_sizes: list):
    """Sweep over both l_max and n_batch."""
    import sys
    import time

    print(f"GPU: {__import__('torch').cuda.get_device_name()}")
    os.environ["MAGIC_DEVICE"] = "cuda"
    sys.path.insert(0, "/root/src")
    import torch
    import shtns
    import ctypes
    import _shtns_cuda as _shtns_mod

    lib = ctypes.CDLL(_shtns_mod.__file__)
    lib.shtns_set_many.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    lib.shtns_set_many.restype = ctypes.c_int

    for lmax in lmax_list:
        os.environ["MAGIC_LMAX"] = str(lmax)
        # Force reimport with new lmax — use importlib
        import importlib
        if 'magic_torch.params' in sys.modules:
            # Clear all magic_torch modules to pick up new MAGIC_LMAX
            mods_to_del = [k for k in sys.modules if k.startswith('magic_torch')]
            for k in mods_to_del:
                del sys.modules[k]
        from magic_torch.config import configure
        configure({"l_max": lmax})
        from magic_torch.params import lm_max, n_theta_max, n_phi_max
        from magic_torch.sht import scal_to_spat, torpol_to_spat
        from magic_torch.precision import CDTYPE, DTYPE, DEVICE

        print(f"\n{'#'*60}")
        print(f"# l_max={lmax}, lm_max={lm_max}, grid={n_theta_max}x{n_phi_max}")
        print(f"{'#'*60}")

        # --- Scalar sweep ---
        print(f"\n  scal_to_spat:")
        print(f"  {'nb':>6} {'bmm':>8} {'shtns':>8} {'spdup':>6}")
        for nb in batch_sizes:
            sh = shtns.sht(lmax, lmax, mres=1, norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE)
            lib.shtns_set_many(ctypes.c_void_p(int(sh.this)), nb, sh.nlm)
            sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                         flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)

            torch.manual_seed(42)
            Slm_t = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
                      + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
            slm_gpu = Slm_t.T.contiguous()
            out_gpu = torch.zeros(nb, sh.nphi, sh.nlat, dtype=DTYPE, device=DEVICE).contiguous()

            n_iter = 30
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            for _ in range(5): scal_to_spat(Slm_t)
            torch.cuda.synchronize()
            s.record()
            for _ in range(n_iter): scal_to_spat(Slm_t)
            e.record()
            torch.cuda.synchronize()
            bmm_ms = s.elapsed_time(e) / n_iter

            torch.cuda.synchronize()
            for _ in range(5): sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
            torch.cuda.synchronize()
            s.record()
            for _ in range(n_iter): sh.cu_SH_to_spat(slm_gpu.data_ptr(), out_gpu.data_ptr())
            e.record()
            torch.cuda.synchronize()
            sht_ms = s.elapsed_time(e) / n_iter

            print(f"  {nb:>6} {bmm_ms:>7.3f} {sht_ms:>7.3f} {bmm_ms/sht_ms:>5.2f}x")

        # --- Vector sweep ---
        print(f"\n  torpol_to_spat:")
        print(f"  {'nb':>6} {'bmm':>8} {'shtns':>8} {'spdup':>6}")
        for nb in batch_sizes:
            sh = shtns.sht(lmax, lmax, mres=1, norm=shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE)
            lib.shtns_set_many(ctypes.c_void_p(int(sh.this)), nb, sh.nlm)
            sh.set_grid(nlat=n_theta_max, nphi=n_phi_max,
                         flags=shtns.SHT_ALLOW_GPU | shtns.SHT_THETA_CONTIGUOUS)

            torch.manual_seed(42)
            Q = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
                 + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
            S = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
                 + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
            T = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
                 + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
            q_g = Q.T.contiguous()
            s_g = S.T.contiguous()
            t_g = T.T.contiguous()
            vr = torch.zeros(nb, sh.nphi, sh.nlat, dtype=DTYPE, device=DEVICE).contiguous()
            vt = torch.zeros_like(vr)
            vp = torch.zeros_like(vr)

            n_iter = 30
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)

            for _ in range(3): torpol_to_spat(Q, S, T)
            torch.cuda.synchronize()
            s_ev.record()
            for _ in range(n_iter): torpol_to_spat(Q, S, T)
            e_ev.record()
            torch.cuda.synchronize()
            bmm_ms = s_ev.elapsed_time(e_ev) / n_iter

            torch.cuda.synchronize()
            for _ in range(3):
                sh.cu_SHqst_to_spat(q_g.data_ptr(), s_g.data_ptr(), t_g.data_ptr(),
                                     vr.data_ptr(), vt.data_ptr(), vp.data_ptr())
            torch.cuda.synchronize()
            s_ev.record()
            for _ in range(n_iter):
                sh.cu_SHqst_to_spat(q_g.data_ptr(), s_g.data_ptr(), t_g.data_ptr(),
                                     vr.data_ptr(), vt.data_ptr(), vp.data_ptr())
            e_ev.record()
            torch.cuda.synchronize()
            sht_ms = s_ev.elapsed_time(e_ev) / n_iter

            print(f"  {nb:>6} {bmm_ms:>7.3f} {sht_ms:>7.3f} {bmm_ms/sht_ms:>5.2f}x")

    print("\nDone!")


@app.local_entrypoint()
def main(lmax: int = 128, n_batch: int = 32, sweep: bool = False, full_sweep: bool = False):
    if full_sweep:
        print(f"Launching full l_max × n_batch sweep, GPU={GPU_TYPE}")
        run_full_sweep.remote(
            [64, 128, 256, 384],
            [4, 32, 128, 512, 1024])
    elif sweep:
        print(f"Launching SHTns batch sweep: l_max={lmax}, GPU={GPU_TYPE}")
        run_batch_sweep.remote(lmax, [1, 4, 16, 32, 64, 128, 256, 512, 769, 1024])
    else:
        print(f"Launching SHTns benchmark: l_max={lmax}, n_batch={n_batch}, GPU={GPU_TYPE}")
        run_benchmark.remote(lmax, n_batch)
