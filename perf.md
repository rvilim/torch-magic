# Performance Log

All timings on CPU (Apple Silicon), single-threaded, dynamo benchmark (l_max=16, n_r_max=33).

## Per-step time

| Timestamp (PDT)        | Change                                      | Time/step | Fortran time/step | Speedup vs prev | Cumulative speedup |
|------------------------|---------------------------------------------|-----------|-------------------|------------------|--------------------|
| 2026-03-28 09:00       | Baseline (sequential radial loops)           | 0.60s     | —                 | —                | 1.0x               |
| 2026-03-28 09:35       | SHT batching over radial levels              | 0.07s     | —                 | 8.5x             | 8.5x               |
| 2026-03-28 10:15       | Batched implicit solvers (precomputed inverse + bmm) | 0.018s    | 0.0035s           | 4.0x             | 34x                |
| 2026-03-28 12:30       | Eliminate SHT m-loops + batch SHT calls in radial_loop | 0.012s | 0.0035s           | 1.5x             | 50x                |
| 2026-03-28 15:00       | Float64 real bmm (view_as_real trick)         | 0.0067s   | 0.0035s           | 1.8x             | 90x                |
| 2026-03-28 17:00       | torch.compile on get_nl (fused element-wise)  | ~14% faster| 0.0035s           | 1.14x            | ~100x              |
| 2026-03-28 18:00       | Pre-allocated velocity buffers                 | 0.0074s   | 0.0035s           | ~1.03x           | ~103x              |
| 2026-03-28 22:00       | MPS (Apple GPU) support + CPU-build init        | see table | —                 | —                | —                  |

## What each change did

### SHT batching (0.60s → 0.07s)
- Eliminated two `for nR in range(33)` Python loops in `radial_loop()`
- Vectorized inner l-mode loop in `torpol_to_spat` (sign-vector matmul trick)
- Vectorized inner l-mode loops in `spat_to_sphertor` (matmul + precomputed `_inv_dLh`)
- Batched `torpol_to_curl_spat` to accept tensor `or2`
- All SHT functions now accept `(lm_max, n_batch)` spectral / `(n_batch, n_theta, n_phi)` grid inputs

### Batched implicit solvers (0.07s → 0.018s)
- Precompute full matrix inverse `A^{-1}` at build time for each l-degree, expand to all lm modes via `st_lm2l` indexing
- Replace `for l in range(l_max+1)` LU solve loops with single `torch.bmm(_inv_all, rhs)` call
- Applied to all four solvers: updateS (N×N), updateZ (N×N), updateB (2× N×N), updateWP (2N×2N)
- WP system: combined inverse includes row+column preconditioning: `fac_col[:, None] * inv(precondA) * fac_row[None, :]`
- l=0 handled via zero inverse rows (solution is zero) or separate scalar solve (WP pressure)

### Eliminate SHT m-loops + batch calls (0.018s → 0.012s)
- Replaced 17-iteration Python m-loop in all 4 core SHT functions with padded batched matmul
- Precompute padded Plm matrices `(n_m_max, NHS, max_nlm)` at module load; single `torch.bmm` per transform
- For `torpol_to_spat`: 6 matrix types stacked into `(6*n_m_max, NHS, max_nlm)`, one bmm call
- For `spat_to_sphertor`: 8 products via `(8*n_m_max, max_nlm, NHS)` stacked bmm
- For `scal_to_SH`/`scal_to_spat`: 2 bmm calls (N/S hemispheres)
- ES/EA parity handled via sign vectors multiplied into precomputed matrices (no ES/EA index gathering)
- In `radial_loop`: batch 4 `torpol_to_spat` calls into 1 (n_batch=128), 3 `scal_to_SH` into 1 (n_batch=93), 3 `spat_to_sphertor` into 1 (n_batch=93)
- Breakdown: radial_loop 7.47ms (62%), lm_loop 4.53ms (38%)

### Float64 real bmm (0.012s → 0.0067s)
- All bmm matrices (Plm, solver inverses) are real-valued but were stored as complex128
- Complex128 bmm uses zgemm (4× the FLOPs of float64 dgemm for a real matrix)
- Use `torch.view_as_real(input)` to get float64 view, float64 bmm, then `view_as_complex(result)`
- For purely imaginary matrices (wPlm_dm in spat_to_sphertor): feed `[-x_imag, x_real]` interleaved
- Applied to: all 4 SHT functions + all 5 implicit solver bmm calls
- Per-call measured speedups: 2.5× (solvers), 5.7× (torpol_to_spat), 2.9× (spat_to_sphertor), 4.0× (WP solver)
- Breakdown: radial_loop 3.95ms (59%), lm_loop 2.69ms (41%)

### torch.compile on get_nl
- `get_nl()` contains 27 element-wise operations on `(33, 24, 48)` float64 grid arrays (Lorentz + advection + entropy + induction)
- These are dispatch-overhead-bound: 27 separate kernel launches for ~38K-element arrays
- `@torch.compile` fuses all operations into a single optimized kernel via Inductor
- Bit-exact: compiled output matches eager mode to the last bit
- Measured: get_nl alone 1.37ms → 0.17ms (8×); full step ~14% faster (radial_loop 7→5ms)
- Note: torch.compile does NOT help complex tensor operations (element-wise ops in solvers are 1.5-3× slower with compile due to poor complex codegen)
- Applied only to get_nl (pure real arithmetic) — complex-heavy functions left eager

## Detailed per-phase comparison (Python vs Fortran)

| Phase | Python | Fortran | Ratio |
|---|---|---|---|
| **Full step** | **7.4ms** | **3.5ms** | **2.1×** |
| radial_loop | 4.6ms | 2.3ms | 2.0× |
| — Inverse SHT | 1.7ms | — | — |
| — get_nl (compiled) | 0.2ms | — | — |
| — Forward SHT | 1.6ms | — | — |
| — get_td + finish_exp | 0.6ms | — | — |
| — vel alloc + scatter | 0.5ms | — | — |
| lm_loop | 2.5ms | 1.1ms | 2.2× |
| — updateS | 0.3ms | — | — |
| — updateZ | 0.3ms | — | — |
| — updateWP | 1.2ms | — | — |
| — updateB | 0.7ms | — | — |
| overhead | 0.3ms | 0.1ms | 5.5× |

The 2× slowdown is uniform across phases — caused by per-tensor-operation dispatch overhead (~3-5μs per op, ~3000-4000 ops per step). No single phase is anomalously slow.

## Fortran reference

Fortran (gfortran-15, single-thread, Apple Silicon): **3.48ms/step** (3.48s for 1000 steps).
- r Loop (SHT + nonlinear): 2.28ms (66%)
- LM Loop (implicit solves): 1.14ms (33%)

Current Python is **~2.1x slower** than Fortran on CPU.

## Projected wall-clock for 1000 steps

| Change              | 1000-step time |
|---------------------|----------------|
| Baseline            | ~10 min        |
| SHT batching        | ~70s           |
| Batched solvers     | ~18s           |
| m-loop elim + batch | ~12s           |
| Float64 real bmm    | ~6.7s          |
| torch.compile get_nl| ~5.8s*         |
| Pre-alloc vel bufs  | ~7.4s          |
| Fortran             | 3.5s           |

*Projected from 14% relative improvement; absolute times vary with system state.

### Pre-allocated velocity buffers (torch.compile → 0.0074s)
- `radial_loop()` allocated 6 `(33, 24, 48)` float64 tensors per step for velocity/vorticity
- Boundaries are always zero (no-slip BCs), bulk is overwritten by torpol_to_spat results
- Pre-allocate at module level, only assign bulk slices — eliminates per-step `torch.zeros` overhead
- Measured: velocity assembly 0.36ms → 0.19ms (47% reduction for that operation)
- Also hoisted `dLh_2d`, `or2_2d`, `or2_bulk_2d` broadcast arrays to module level

### MPS (Apple GPU) support + resolution sweep

Enabled MPS by building all initialization tensors on CPU (avoiding GPU→CPU sync from scalar Python loops). Also chunked the batched solver bmm to store only `(l_max+1, N, N)` unique per-l inverses instead of `(lm_max, N, N)`.

**Resolution sweep results (2026-03-29, Apple M-series, single-thread Fortran, CNAB2 insulating IC):**

Steady-state ms/step (excludes first step with matrix build). Fortran: gfortran-15 serial. CPU: PyTorch f64. MPS: PyTorch f32.

| l_max | n_r | lm_max | grid      | Fortran ms | CPU f64 ms | MPS f32 ms | CPU vs Fortran | MPS vs Fortran | MPS vs CPU |
|-------|-----|--------|-----------|-----------|-----------|------------|----------------|----------------|------------|
| 16    | 33  | 153    | 24×48     | 10.8      | 8.1       | 15.2       | 1.3×           | 0.7×           | 0.5×       |
| 32    | 65  | 561    | 48×96     | 34.6      | 36.7      | 16.3       | 0.9×           | **2.1×**       | **2.3×**   |
| 64    | 129 | 2145   | 96×192    | 399       | 279       | 55.2       | **1.4×**       | **7.2×**       | **5.1×**   |
| 128   | 257 | 8385   | 192×384   | 5749      | 2833      | 411        | **2.0×**       | **14.0×**      | **6.9×**   |

Key observations:
- MPS crossover vs CPU at l~28, vs Fortran at l~24
- CPU beats Fortran at all resolutions except l=32 (dispatch overhead vs BLAS efficiency)
- At l=128: MPS is 14× faster than Fortran, 6.9× faster than CPU
- MPS scales much better than CPU/Fortran (GPU parallelism dominates at larger sizes)

### doubleDiffusion benchmark (2026-03-29)

BPR353 DIRK (4-stage, 3 with radial_loop) + composition field (xi). No magnetic field (mode=1).
l_max=64, n_r_max=33, minc=4, n_cheb_max=31. Restart from saturated checkpoint.

| Backend | ms/step | vs Fortran |
|---------|---------|------------|
| Fortran (gfortran-15, serial) | 52.9 | 1.0× |
| Python CPU (f64) | 56.4 | 0.94× (1.07× slower) |
| Python MPS (f32) baseline | 53.7 | 0.98× (1.01× slower) |
| Python MPS (f32) optimized | 21.6 | **2.45×** faster |

Energy comparison (1000 steps):
- e_kin_pol max rel err: 2.4e-10
- e_kin_tor max rel err: 4.1e-9
- e_mag: exactly zero (mode=1, no magnetic field)
- Errors constant over 1000 steps (steady state, not growing)

Note: BPR353 has 3 radial_loop stages per step vs 1 for CNAB2, so per-step time is ~3× the CNAB2 cost at same resolution. The l=64 CNAB2 time is ~279ms/step (from resolution sweep above, n_r=129), while DD BPR353 at l=64 is only 56ms/step because n_r=33 (much smaller radial grid).

### doubleDiffusion MPS optimization (2026-03-29)

Fused lm_loop pipeline + solver cache optimizations for MPS dispatch overhead reduction:

| Change | DD MPS ms/step | vs previous |
|--------|---------------|-------------|
| Baseline (sequential solvers) | 53.7 | — |
| Mega-tensor views (cat-free RHS) | ~50 | ~7% |
| Fused lm_loop + deferred costf + unified D123 | 33.9 | 1.5× |
| GPU-resident pack indices + pre-allocated solver buffers | 27.0 | 1.25× |
| p0Mat GPU inverse + in-place add + precomputed constants | 21.6 | 1.25× |

Final: **21.6ms/step MPS** vs 52.9ms Fortran = **2.45× faster than Fortran**, 2.6× faster than Python CPU (57ms).

Per-stage breakdown (BPR353, 4 stages avg):
- radial_loop: 2.72ms/call × 3 stages = 8.2ms
- lm_loop: 3.70ms/call × 4 stages = 14.8ms
- overhead: ~0.6ms
- Total: ~21.6ms/step

### Anelastic benchmark (hydro_bench_anel_lowres) (2026-03-29)

Anelastic hydrodynamic benchmark: strat=5 (polytropic), polind=2, no magnetic field (mode=1).
l_max=16, n_r_max=33, minc=1, CNAB2 time scheme. Non-curl advection (11 grid-space fields).

| Backend | ms/step | vs Fortran |
|---------|---------|------------|
| Fortran (gfortran-15, serial) | 2.86 | 1.0× |
| Python CPU (f64) | 7.23 | 0.40× (2.5× slower) |
| Python MPS (f32) | 10.71 | 0.27× (3.7× slower) |

Notes:
- 200 steps measured, excluding first step (matrix build + JIT warmup)
- Anelastic adds density weighting (`orho1`, `beta`, variable `temp0`) and non-curl advection (11 grid-space fields vs 6)
- No magnetic field: updateB disabled, so lm_loop has only S+Z+WP solvers
- MPS `torch.compile` disabled for `get_nl_anel` (too many Metal shader arguments)
- At l=16 the problem is too small for GPU — dispatch overhead dominates
- CPU 2.5× slower than Fortran (vs 2.1× for Boussinesq dynamo), consistent with dispatch overhead
- Fortran anelastic is faster per-step than Boussinesq dynamo (2.86 vs 3.48 ms) because no magnetic field
- MPS crossover vs Fortran expected at l~24-32 (same as Boussinesq, see resolution sweep above)

### Packed BMM solver (2025-03-28)
Replaced expand-and-bmm (`inv[l_index]` → `(lm_max, N, N)` bmm) with packed approach:
sort RHS by l → scatter to `(L, max_m, N, 2)` → reshape → single `(L, N, N) @ (L, N, max_m*2)` bmm → gather back.
All pack/unpack indices precomputed and cached. No Python loops at solve time.
- l=64 MPS: 89ms → 55ms (lm_loop 1.6×)
- l=128 MPS: 975ms → 405ms (lm_loop 6.5×: 675ms → 104ms)
- l=128 profile: radial_loop 303ms (SHT 252ms + get_nl 17ms), lm_loop 104ms

### Full-resolution anelastic (hydro_bench_anel, l=143, n_r=97) (2026-03-29)

Anelastic hydrodynamic benchmark at full resolution: strat=5, polind=2, mode=1 (no mag).
l_max=143, n_r_max=97, n_cheb_max=95, n_theta=216, n_phi=432, lm_max=10440.
CNAB2 time scheme, AM corrections enabled (l_correct_AMz, l_correct_AMe).

| Backend | ms/step | Notes |
|---------|---------|-------|
| Python CPU (f64) | ~1011 | 300 steps, matches Fortran reference.out |

| Fortran (gfortran-15, serial) | 2222 | Baseline |
| Python CPU (f64) | 991 | **2.2× faster than Fortran** |
| Python MPS (f32) | 276 | **8.1× faster than Fortran** |

20 steps measured, excluding first step (matrix build + JIT warmup).
Fortran timing from MagIC log "Mean wall time for one pure time step".

Key observations:
- CPU is now **faster** than Fortran at this resolution (BLAS efficiency dominates dispatch overhead)
- MPS is 8.1× faster than Fortran — GPU parallelism fully utilized at this problem size
- MPS is 3.6× faster than CPU
- Fortran r-loop dominates: 2024ms (91%) — SHT at high l is compute-bound
- Fortran breakdown: Spec→Spat 1274ms, Spat→Spec 685ms, LM solves 112ms

Energy validation (300 steps at l=143, compared to Fortran reference.out):
- e_kin_pol max rel err: 4.1e-9
- e_kin_tor max rel err: 4.6e-9
- Agreement to ~9 significant digits over 300 steps

### H100 CUDA dense solver optimization (2026-04-05)

Eliminated sequential kernel launches in FD solvers on CUDA GPU (N ≤ 1024).

**Problem**: At l_max=64, N=129 on H100, lm_loop was 77ms/step (92% of total 84ms).
Sequential Thomas/pentadiag sweeps launched one kernel per radial point, starving the GPU.

**Three fixes**:
1. **Dense inverse for s/z/xi/b solvers**: precompute `A^{-1}` at build time, use
   `chunked_solve_complex` (single batched bmm) instead of sequential Thomas/pentadiag.
   Condition numbers ~O(N²) — safe for inverse up to N=1024.
2. **Dense inverse for w (double-curl) solver**: same approach. Despite 4th-order operator,
   condition numbers are at most ~7e9 at N=1024 (still ~6 digits in float64).
   Previous LU factorization approach was 14ms on H100 vs 0.12ms with inverse.
3. **p0Mat dense inverse**: the l=0 pressure tridiagonal solver used `solve_tridiag_real`
   with a Python for-loop and `.item()` calls — each `.item()` forces a CPU-GPU sync (~100μs).
   129 syncs × 100μs = 13ms. Fixed by precomputing p0Mat inverse on GPU.

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| lm_loop | 77ms | 4.4ms | 17.5× |
| updateW | 18ms | 1.9ms | 9.5× |
| updateS | 0.8ms | 0.7ms | — |
| updateZ | 0.7ms | 0.6ms | — |
| updateB | 1.4ms | 1.2ms | — |
| radial_loop | 5.7ms | 5.2ms | — |
| **Total step** | **84ms** | **9.9ms** | **8.5×** |

Bottleneck shifted: lm_loop (92% → 45%), radial_loop (7% → 53%).
Radial loop now dominated by SHT matmuls (torpol_to_spat 1.4ms, spat_to_sphertor 1.1ms).

Memory: ~43MB for 5 solver inverses at N=129 (65×129×129 float64 each).
Gate: N ≤ 1024. For N > 1024, falls back to batched Thomas/pentadiag.

### B200 profiling (2026-04-05)

CUDA event-based profiling on NVIDIA B200 (192 GB HBM3e, 72 TFLOP/s fp64).
200 steps, first 2 skipped as warmup.

#### l=128 (N=257, no chunking) — 56 ms/step

| Component | ms/step | % |
|---|---|---|
| **radial_loop total** | **18.5** | **33%** |
| radial_loop.inv_sht | 8.4 | 15% |
| radial_loop.fwd_sht | 7.5 | 13% |
| radial_loop.get_td | 1.5 | 3% |
| radial_loop.get_nl | 0.6 | 1% |
| **lm_loop total** | **7.8** | **14%** |
| lm_loop.solve_szw | 5.4 | 10% |
| lm_loop.d123_matmul | 1.8 | 3% |
| lm_loop.solve_b | 2.4 | 4% |
| **Unaccounted (output, alloc, bookkeeping)** | **~30** | **~53%** |

#### l=384 (N=769, chunk_size=32) — 2122 ms/step (corrected profiler)

| Component | ms/step | % |
|---|---|---|
| **radial_loop total** | **633** | **30%** |
| radial_loop.inv_sht | 277 | 13% |
| radial_loop.fwd_sht | 285 | 13% |
| radial_loop.get_td | 44 | 2% |
| radial_loop.get_nl | 16 | 1% |
| **lm_loop total** | **330** | **16%** |
| lm_loop.solve_szw | 240 | 11% |
| lm_loop.d123_matmul | 127 | 6% |
| lm_loop.solve_b | 90 | 4% |
| lm_loop.wp_solve | 49 | 2% |
| lm_loop.scalar_solve | 33 | 2% |
Key observations:
- **SHT dominates**: inv_sht (277ms) + fwd_sht (285ms) = 562ms
- **SHT at l=384**: 50% of bmm FLOPs wasted on zero-padded Plm entries
- **get_nl**: only 16ms — torch.compile fusion working well
- Note: profiler TOTAL was double-counting (summing outer + inner timers). Fixed in v3.
  Real step time = sum of top-level timers only (radial_loop + lm_loop) ≈ 965 ms.

#### l=384 after view_as_real D123 + rotate_imex fix (2026-04-05)

Split real/imag DGEMM for all derivative matmuls (D123, get_dr, get_ddr, get_dddr).
Removed .clone() from rotate_imex (non-overlapping dim-2 slices don't need it).

| Component | Before | After | Change |
|---|---|---|---|
| **radial_loop** | **633** | **626** | -1% |
| radial_loop.inv_sht | 277 | 278 | — |
| radial_loop.fwd_sht | 285 | 285 | — |
| radial_loop.get_td | 44 | 36 | -18% |
| radial_loop.get_nl | 16 | 16 | — |
| **lm_loop** | **330** | **253** | **-23%** |
| lm_loop.solve_szw | 240 | 181 | -25% |
| lm_loop.d123_matmul | 127 | 70 | **-45%** |
| lm_loop.solve_b | 90 | 72 | **-20%** |
| lm_loop.wp_solve | 49 | 49 | — |
| lm_loop.scalar_solve | 33 | 33 | — |
| **TOTAL (top-level)** | **~965** | **880** | **-9%** |

SHT now 64% of step time (inv_sht 278 + fwd_sht 285 = 563 ms).
50% of SHT bmm FLOPs are wasted on zero-padded Plm entries — next optimization target.

#### l=384 after bucketed SHT (K=4 buckets) (2026-04-05)

Split m-modes into 4 buckets by nlm size to reduce padding waste from 50% to ~20%.
Each bucket has its own Plm matrices padded to local max_nlm.

| Component | Before | After | Change |
|---|---|---|---|
| **radial_loop** | **626** | **545** | **-13%** |
| radial_loop.inv_sht | 278 | 239 | **-14%** |
| radial_loop.fwd_sht | 285 | 241 | **-15%** |
| radial_loop.get_td | 36 | 36 | — |
| radial_loop.get_nl | 16 | 16 | — |
| **lm_loop** | **253** | **254** | — |
| **TOTAL (top-level)** | **880** | **798** | **-9%** |

SHT: 563 → 480 ms (-15%). Total: 880 → 798 ms (-9%).
Savings less than the 37% FLOP reduction because low-m buckets (largest matrices)
are compute-bound — GPU wasn't fully utilizing the padding zeros anyway.
Polar optimization (per-bucket NHS truncation) is the next target.

#### l=384 after polar optimization (2026-04-05)

Per-bucket NHS truncation: skip polar theta rows where Plm < 1e-14.
Bucket 0: skip 0/288, Bucket 1: skip 27, Bucket 2: skip 69, Bucket 3: skip 121.

| Component | Before | After | Change |
|---|---|---|---|
| radial_loop.inv_sht | 239 | 225 | -6% |
| radial_loop.fwd_sht | 241 | 227 | -6% |
| **TOTAL (top-level)** | **798** | **768** | **-4%** |

SHT: 480 → 452 ms (-6%). Modest gain — high-m buckets benefit most but have
fewest modes. Cumulative: 965 → 768 ms (**-20% from baseline**).

#### l=384 chunk size analysis (2026-04-05)

At l=384, spectral fields + solver inverses + SHT matrices consume ~160 GB on B200 (178 GB
usable). This leaves ~18 GB for grid-space temporaries during each chunk iteration.

torpol_to_spat batches 4 field groups (B, curlB, V, curlV) so n_batch = 4 × chunk_size.
The irfft output is 3 × (n_theta × n_phi × n_batch × 8 bytes). Memory per chunk:
- C=32: ~1.6 GB grid temps → OK (current, 25 chunks, 768 ms/step)
- C=64: ~3.2 GB → OK (12 chunks, profile pending)
- C=128: ~6.4 GB → borderline (~18 GB headroom)
- C=256: ~12.8 GB → OOM (tried, failed at irfft needing 5 GB with only 4.5 GB free)

Investigation: critique agent suggested kernel launch overhead might dominate.
Result: **chunk=64 gives 765 ms vs chunk=32's 768 ms — no improvement.**

| chunk_size | Chunks | SHT (inv+fwd) | Total |
|---|---|---|---|
| 32 | 25 | 452 ms | 768 ms |
| 64 | 12 | 450 ms | 765 ms |
| 128 | — | not tested (borderline OOM) | — |
| 256 | — | OOM | — |

**Conclusion: kernel launch count is NOT the bottleneck.** The SHT cost is compute/bandwidth-bound
in the bmm itself, not launch overhead. This means CUTLASS grouped GEMM (which mainly reduces
launches) would also have limited impact. The path to further SHT speedup requires changing the
algorithm — either Triton on-the-fly recurrence or an external SHT library (SHTns).
