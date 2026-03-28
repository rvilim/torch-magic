#!/usr/bin/env python3
"""Resolution sweep benchmark: Python (CPU / MPS) vs Fortran.

Runs each configuration in a subprocess (params are module-level, so
different resolution = different process).

Usage:
    cd magic-torch && uv run python benchmark.py
    cd magic-torch && uv run python benchmark.py --lmax 16,32
    cd magic-torch && uv run python benchmark.py --steps 50 --no-fortran
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time


MAGIC_EXE = os.path.join(os.path.dirname(__file__), "..", "src", "magic.exe")

# Matches truncation.f90 prime_decomposition
def _prime_decomposition(n):
    best = None
    for i in range(13):
        for j in range(7):
            for k in range(7):
                val = 2**i * 3**j * 5**k
                if val >= n and (best is None or val < best):
                    best = val
    return best


def grid_params(l_max, n_r=None):
    """Derive all grid params for a given l_max (matching truncation.f90)."""
    nalias = 20
    minc = 1
    m_max = l_max
    if n_r is None:
        n_r = 2 * l_max + 1
    n_phi_tot = _prime_decomposition(2 * ((30 * l_max) // nalias))
    n_theta = n_phi_tot // 2
    n_phi = n_phi_tot // minc
    n_m = m_max // minc + 1
    lm = (l_max + 1) * (l_max + 2) // 2
    n_r_ic = (n_r + 1) // 2
    return {
        "l_max": l_max, "m_max": m_max, "n_r_max": n_r,
        "n_cheb_max": n_r, "n_r_ic_max": n_r_ic, "n_cheb_ic_max": n_r_ic,
        "n_theta_max": n_theta, "n_phi_max": n_phi,
        "n_m_max": n_m, "lm_max": lm,
    }


# ---------------------------------------------------------------------------
# Python runner (subprocess)
# ---------------------------------------------------------------------------

_PY_RUNNER = """\
import time, json, sys
from magic_torch.init_fields import initialize_fields
from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
from magic_torch.params import dtmax, l_max, n_r_max, lm_max, n_theta_max, n_phi_max

N = int(sys.argv[1])
WARMUP = int(sys.argv[2])

initialize_fields()
setup_initial_state()
initialize_dt(dtmax)

# Warmup steps (not timed)
for n in range(1, WARMUP + 1):
    one_step(n, dtmax)

import torch
if torch.cuda.is_available():
    torch.cuda.synchronize()
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.mps.synchronize()

t0 = time.perf_counter()
for n in range(WARMUP + 1, WARMUP + N + 1):
    one_step(n, dtmax)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    torch.mps.synchronize()

elapsed = time.perf_counter() - t0
ms_per_step = 1000 * elapsed / N

from magic_torch.precision import DEVICE, DTYPE
result = {
    "ms_per_step": ms_per_step,
    "device": str(DEVICE),
    "dtype": str(DTYPE),
    "l_max": l_max,
    "n_r_max": n_r_max,
    "lm_max": lm_max,
    "n_theta_max": n_theta_max,
    "n_phi_max": n_phi_max,
    "n_steps": N,
}
print("BENCH_RESULT:" + json.dumps(result))
"""


def run_python(l_max, device, n_steps, warmup, n_r=None):
    """Run Python benchmark in subprocess, return ms/step or None on failure."""
    env = os.environ.copy()
    env["MAGIC_LMAX"] = str(l_max)
    env["MAGIC_DEVICE"] = device
    if n_r is not None:
        env["MAGIC_NR"] = str(n_r)

    src_dir = os.path.join(os.path.dirname(__file__), "src")
    cmd = [sys.executable, "-c", _PY_RUNNER, str(n_steps), str(warmup)]
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env,
            cwd=os.path.dirname(__file__),
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"

    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            data = json.loads(line[len("BENCH_RESULT:"):])
            return data["ms_per_step"], data

    return None, result.stderr[-500:] if result.stderr else "no output"


# ---------------------------------------------------------------------------
# Fortran runner
# ---------------------------------------------------------------------------

_INPUT_NML_TEMPLATE = """\
&grid
 n_r_max     ={n_r_max},
 n_cheb_max  ={n_cheb_max},
 l_max       ={l_max},
 m_max       ={m_max},
 n_r_ic_max  ={n_r_ic_max},
 n_cheb_ic_max={n_cheb_ic_max},
 nalias      =20,
 minc        =1,
/
&control
 mode        =0,
 tag         ="bench",
 n_time_steps={n_time_steps},
 courfac     =2.5D0,
 alffac      =1.0D0,
 dtmax       =1.0D-4,
 alpha       =0.6D0,
 runHours    =12,
 runMinutes  =00,
 anelastic_flavour='ENT',
/
&phys_param
 ra          =1.0D5,
 ek          =1.0D-3,
 pr          =1.0D0,
 prmag       =5.0D0,
 radratio    =0.35D0,
 ktops       =1,
 kbots       =1,
 ktopv       =2,
 kbotv       =2,
/
&start_field
 l_start_file=.false.,
 start_file  ="None",
 init_b1     =3,
 amp_b1      =5,
 init_s1     =0404,
 amp_s1      =0.1,
/
&output_control
 n_log_step  =1,
 n_graphs    =0,
 n_rsts      =0,
 n_stores    =0,
 runid       ="Benchmark sweep",
 l_movie     =.false.,
 l_RMS       =.false.,
/
&mantle
 nRotMa      =0
/
&inner_core
 sigma_ratio =0.d0,
 nRotIC      =0,
/
"""


def run_fortran(l_max, n_steps, n_r=None):
    """Run Fortran benchmark, return ms/step or None."""
    exe = os.path.abspath(MAGIC_EXE)
    if not os.path.isfile(exe):
        return None, f"Fortran binary not found at {exe}"

    gp = grid_params(l_max, n_r)
    # Fortran n_time_steps includes the initial output step
    nml = _INPUT_NML_TEMPLATE.format(
        n_time_steps=n_steps + 1,
        **gp,
    )

    with tempfile.TemporaryDirectory(prefix="magic_bench_") as tmpdir:
        nml_path = os.path.join(tmpdir, "input.nml")
        with open(nml_path, "w") as f:
            f.write(nml)
        # The modified Fortran binary has dump_arrays that writes here
        os.makedirs(os.path.join(tmpdir, "fortran_dumps"), exist_ok=True)

        try:
            result = subprocess.run(
                [exe, "input.nml"],
                capture_output=True, text=True, timeout=600,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"

        # Parse log.bench for wall time
        log_path = os.path.join(tmpdir, "log.bench")
        if not os.path.isfile(log_path):
            return None, f"no log file; stderr: {result.stderr[-300:]}"

        with open(log_path) as f:
            log_text = f.read()

        # Look for the last "Mean wall time for time step" line
        pattern = r"Mean wall time for one time step\s*:\s*([\d.E+-]+)"
        matches = re.findall(pattern, log_text)
        if matches:
            ms = float(matches[-1]) * 1000
            return ms, {"ms_per_step": ms, "l_max": l_max, "n_r_max": gp["n_r_max"]}

        # Fallback: look for per-step lines
        pattern2 = r"Mean wall time for time step:\s*([\d.E+-]+)"
        matches2 = re.findall(pattern2, log_text)
        if matches2:
            ms = float(matches2[-1]) * 1000
            return ms, {"ms_per_step": ms, "l_max": l_max, "n_r_max": gp["n_r_max"]}

        return None, "could not parse wall time from log"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resolution sweep benchmark")
    parser.add_argument("--lmax", default="16,32,64,128",
                        help="Comma-separated l_max values (default: 16,32,64,128)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of timed steps (default: 20)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup steps before timing (default: 5)")
    parser.add_argument("--no-fortran", action="store_true",
                        help="Skip Fortran runs")
    parser.add_argument("--no-mps", action="store_true",
                        help="Skip MPS runs")
    parser.add_argument("--no-cpu", action="store_true",
                        help="Skip CPU runs")
    args = parser.parse_args()

    lmax_list = [int(x.strip()) for x in args.lmax.split(",")]

    print(f"Benchmark: {args.steps} timed steps, {args.warmup} warmup steps")
    print(f"Resolutions: l_max = {lmax_list}")
    print()

    results = []

    for lm in lmax_list:
        gp = grid_params(lm)
        row = {
            "l_max": lm,
            "n_r": gp["n_r_max"],
            "lm_max": gp["lm_max"],
            "n_theta": gp["n_theta_max"],
            "n_phi": gp["n_phi_max"],
        }

        # Python CPU
        if not args.no_cpu:
            print(f"[l_max={lm}] Python CPU ... ", end="", flush=True)
            ms, info = run_python(lm, "cpu", args.steps, args.warmup)
            if ms is not None:
                print(f"{ms:.2f} ms/step")
            else:
                print(f"FAILED: {info}")
            row["cpu_ms"] = ms

        # Python MPS
        if not args.no_mps:
            print(f"[l_max={lm}] Python MPS ... ", end="", flush=True)
            ms, info = run_python(lm, "mps", args.steps, args.warmup)
            if ms is not None:
                print(f"{ms:.2f} ms/step")
            else:
                print(f"FAILED: {info}")
            row["mps_ms"] = ms

        # Fortran
        if not args.no_fortran:
            print(f"[l_max={lm}] Fortran    ... ", end="", flush=True)
            ms, info = run_fortran(lm, args.steps)
            if ms is not None:
                print(f"{ms:.2f} ms/step")
            else:
                print(f"FAILED: {info}")
            row["fortran_ms"] = ms

        results.append(row)
        print()

    # Summary table
    print("=" * 90)
    print(f"{'l_max':>5} {'n_r':>5} {'lm_max':>7} {'grid':>12}"
          f"  {'CPU ms':>10} {'MPS ms':>10} {'Fortran ms':>12}")
    print("-" * 90)
    for r in results:
        grid_str = f"{r['n_theta']}x{r['n_phi']}"
        cpu = f"{r.get('cpu_ms', 0):.2f}" if r.get("cpu_ms") else "---"
        mps = f"{r.get('mps_ms', 0):.2f}" if r.get("mps_ms") else "---"
        fortran = f"{r.get('fortran_ms', 0):.2f}" if r.get("fortran_ms") else "---"
        print(f"{r['l_max']:>5} {r['n_r']:>5} {r['lm_max']:>7} {grid_str:>12}"
              f"  {cpu:>10} {mps:>10} {fortran:>12}")
    print("=" * 90)


if __name__ == "__main__":
    main()
