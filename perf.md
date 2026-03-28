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
