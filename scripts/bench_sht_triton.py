#!/usr/bin/env python3
"""Benchmark: Triton on-the-fly SHT vs batched bmm.

Runs on Modal with GPU. Compares scal_to_spat implementations for
correctness and performance.

Usage:
    uv run modal run scripts/bench_sht_triton.py -- --lmax 128
    uv run modal run scripts/bench_sht_triton.py -- --lmax 384 --n_batch 64
    MAGIC_MODAL_GPU=B200 uv run modal run scripts/bench_sht_triton.py -- --lmax 384
"""

import os

import modal

GPU_TYPE = os.environ.get("MAGIC_MODAL_GPU", "H100")

app = modal.App("sht-triton-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "triton")
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def run_benchmark(lmax: int, n_batch_override: int, n_iter: int, n_warmup: int):
    import sys
    import time

    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_DEVICE"] = "cuda"
    os.environ["MAGIC_POLAR_OPT"] = "true"
    sys.path.insert(0, "/root/src")

    import torch
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton version: {__import__('triton').__version__}")

    t_import_start = time.perf_counter()
    from magic_torch.params import lm_max, n_r_max, n_m_max, n_theta_max, n_phi_max, l_max
    from magic_torch.sht import scal_to_spat
    from magic_torch.sht_triton import scal_to_spat_triton, _NHS
    from magic_torch.precision import CDTYPE, DTYPE, DEVICE
    t_import = time.perf_counter() - t_import_start
    print(f"Import time: {t_import:.1f}s (Plm build + module init)")

    n_batch = n_batch_override if n_batch_override > 0 else n_r_max
    print(f"\nl_max={l_max}, lm_max={lm_max}, n_m_max={n_m_max}")
    print(f"NHS={_NHS}, n_theta={n_theta_max}, n_phi={n_phi_max}")
    print(f"n_batch={n_batch}, n_iter={n_iter}, n_warmup={n_warmup}")

    # Random spectral coefficients
    torch.manual_seed(42)
    Slm = (torch.randn(lm_max, n_batch, dtype=DTYPE, device=DEVICE)
           + 1j * torch.randn(lm_max, n_batch, dtype=DTYPE, device=DEVICE)).to(CDTYPE)

    # === Correctness ===
    print("\n--- Correctness ---")
    ref = scal_to_spat(Slm)
    out = scal_to_spat_triton(Slm)
    abs_err = (ref - out).abs().max().item()
    rel_err = abs_err / ref.abs().max().item() if ref.abs().max().item() > 0 else 0
    print(f"Max absolute error: {abs_err:.2e}")
    print(f"Max relative error: {rel_err:.2e}")
    print(f"Output range: [{ref.min().item():.2e}, {ref.max().item():.2e}]")

    # Also compare against non-bucketed path as gold reference
    import magic_torch.sht as sht
    old_bucketed = sht._USE_BUCKETED
    sht._USE_BUCKETED = False
    gold = scal_to_spat(Slm)
    sht._USE_BUCKETED = old_bucketed

    abs_err_gold = (out - gold).abs().max().item()
    rel_err_gold = abs_err_gold / gold.abs().max().item() if gold.abs().max().item() > 0 else 0
    print(f"Triton vs gold (non-bucketed): abs={abs_err_gold:.2e}, rel={rel_err_gold:.2e}")
    abs_err_bmm_gold = (ref - gold).abs().max().item()
    print(f"Bucketed bmm vs gold: abs={abs_err_bmm_gold:.2e}")

    # === Timing ===
    print(f"\n--- Timing ({n_warmup} warmup, {n_iter} measured) ---")

    results = {}
    for name, fn in [("bmm (bucketed+polar)", scal_to_spat),
                      ("triton", scal_to_spat_triton)]:
        # Warmup (includes Triton JIT on first call)
        t_warmup_start = time.perf_counter()
        for _ in range(n_warmup):
            _ = fn(Slm)
        torch.cuda.synchronize()
        t_warmup = time.perf_counter() - t_warmup_start
        print(f"  {name} warmup: {t_warmup:.1f}s ({n_warmup} calls)")

        # Timed run
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(n_iter):
            _ = fn(Slm)
        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev) / n_iter
        results[name] = ms
        print(f"  {name}: {ms:.3f} ms/call")

    # Speedup
    if len(results) == 2:
        bmm_ms = list(results.values())[0]
        tri_ms = list(results.values())[1]
        print(f"\nSpeedup: {bmm_ms / tri_ms:.2f}x")

    # FLOP estimate
    total_fma = sum(l_max - m + 1 for m in range(0, l_max + 1, 1))  # = lm_max
    total_flops = total_fma * _NHS * n_batch * 2 * 2  # 2 hemispheres, 2 FMA per step
    gflops = total_flops / 1e9
    for name, ms in results.items():
        tflops = gflops / ms
        print(f"  {name}: {tflops:.1f} GFLOP/s ({gflops:.1f} GFLOP)")

    print("\nDone!")


@app.local_entrypoint()
def main(lmax: int = 128, n_batch: int = 0, n_iter: int = 50, n_warmup: int = 5):
    print(f"Launching SHT Triton benchmark: l_max={lmax}, GPU={GPU_TYPE}")
    run_benchmark.remote(lmax, n_batch, n_iter, n_warmup)
