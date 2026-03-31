"""Compare Python vs Fortran per-stage fields for boussBenchSat."""
import os
os.environ["MAGIC_TIME_SCHEME"] = "BPR353"
os.environ["MAGIC_LMAX"] = "64"
os.environ["MAGIC_NR"] = "33"
os.environ["MAGIC_MINC"] = "4"
os.environ["MAGIC_NCHEBMAX"] = "31"
os.environ["MAGIC_NCHEBICMAX"] = "15"
os.environ["MAGIC_RA"] = "1.1e5"
os.environ["MAGIC_SIGMA_RATIO"] = "1.0"
os.environ["MAGIC_KBOTB"] = "3"
os.environ["MAGIC_NROTIC"] = "1"
os.environ["MAGIC_DEVICE"] = "cpu"

import sys
import struct
import numpy as np
import torch
from pathlib import Path

# Load Fortran binary dump
def load_fortran_dat(path):
    """Load a Fortran binary dump file (stream access)."""
    with open(path, 'rb') as f:
        ndim = struct.unpack('>i', f.read(4))[0]
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('>i', f.read(4))[0])
        count = 1
        for s in shape:
            count *= s
        if ndim == 0:
            # Scalar
            data = np.frombuffer(f.read(8), dtype='>f8').astype(np.float64)
            return data
        data = np.frombuffer(f.read(count * 8), dtype='>f8').astype(np.float64)
        # Fortran column-major to C row-major
        if ndim == 2:
            data = data.reshape(shape[1], shape[0]).T
        elif ndim == 1:
            data = data.reshape(shape)
    return data

def load_fortran_complex(name, dump_dir):
    """Load a complex Fortran dump (re + im parts)."""
    re = load_fortran_dat(os.path.join(dump_dir, f'{name}_re.dat'))
    im = load_fortran_dat(os.path.join(dump_dir, f'{name}_im.dat'))
    return re + 1j * im

# Paths
BOUSS_DIR = Path(__file__).parent.parent.parent / 'samples' / 'boussBenchSat'
DUMP_DIR = str(BOUSS_DIR / 'fortran_dumps')
CKPT = str(BOUSS_DIR / 'checkpoint_end.start')

# Run Python to get per-stage data
from magic_torch.main import load_fortran_checkpoint
from magic_torch.step_time import setup_initial_state, initialize_dt, radial_loop, lm_loop, build_all_matrices
from magic_torch import fields, dt_fields
from magic_torch.params import n_r_max, n_r_ic_max, lm_max
from magic_torch.time_scheme import tscheme

sim_time = load_fortran_checkpoint(CKPT)

# Print z10 at ICB BEFORE setup_initial_state
from magic_torch.blocking import st_lm2
_l1m0 = st_lm2[1, 0].item()
print(f"z10_icb BEFORE setup = {fields.z_LMloc[_l1m0, n_r_max - 1].real.item():.16e}")

setup_initial_state()

# Print z10 at ICB AFTER setup_initial_state (should be same)
print(f"z10_icb AFTER  setup = {fields.z_LMloc[_l1m0, n_r_max - 1].real.item():.16e}")

from magic_torch.pre_calculations import c_dt_z10_ic
print(f"c_dt_z10_ic = {c_dt_z10_ic:.16e}")
print(f"domega_ic_dt.old[0] = {dt_fields.domega_ic_dt.old[0]:.16e}")

initialize_dt(2.0e-4)

tscheme.dt[0] = 2.0e-4
tscheme.set_weights()
build_all_matrices()

# Snake-to-standard permutation for boussBenchSat
sys.path.insert(0, str(Path(__file__).parent))
from conftest import _compute_snake_to_standard_perm
snake2st = _compute_snake_to_standard_perm(l_max=64, minc=4).numpy()

def reorder_snake_to_st(arr):
    """Reorder from Fortran snake LM to standard ordering."""
    result = np.zeros_like(arr)
    result[snake2st] = arr
    return result

print("=" * 70)
print("Stage-by-stage comparison: Python vs Fortran")
print("=" * 70)

from magic_torch.blocking import st_lm2

for stage in range(1, 5):
    tscheme.istage = stage
    print(f"\n  [TRACE] stage={stage} BEFORE radial_loop: old[0]={dt_fields.domega_ic_dt.old[0]:.16e}")

    # Explicit terms (skip for stage 4)
    if tscheme.l_exp_calc[stage - 1]:
        radial_loop()
    print(f"  [TRACE] stage={stage} AFTER  radial_loop: old[0]={dt_fields.domega_ic_dt.old[0]:.16e}")

    # Print domega_ic_dt IMEX components BEFORE solve at stage 2
    if stage == 2:
        d = dt_fields
        print(f"\n=== domega_ic_dt IMEX components BEFORE stage 2 solve ===")
        print(f"  old[0]   = {d.domega_ic_dt.old[0]:.16e}")
        for j in range(stage):
            print(f"  expl[{j}]  = {d.domega_ic_dt.expl[j]:.16e}")
        for j in range(stage):
            print(f"  impl[{j}]  = {d.domega_ic_dt.impl[j]:.16e}")
        # Compute dom_ic manually
        dom_ic = tscheme.set_imex_rhs_scalar(d.domega_ic_dt)
        print(f"  dom_ic (assembled) = {dom_ic:.16e}")
        # Print Butcher weights
        for j in range(stage):
            print(f"  butcher_exp[{stage},{j}] = {tscheme.butcher_exp[stage,j].item():.16e}")
        for j in range(stage):
            print(f"  butcher_imp[{stage},{j}] = {tscheme.butcher_imp[stage,j].item():.16e}")

        # Also print dzdt IMEX components for l=1,m=0 at ICB
        l1m0 = st_lm2[1, 0].item()
        N = n_r_max
        print(f"\n=== dzdt IMEX for l=1,m=0 at ICB (n_r={N-1}) ===")
        print(f"  old[0]   = {d.dzdt.old[l1m0, N-1, 0].item()}")
        for j in range(stage):
            print(f"  expl[{j}]  = {d.dzdt.expl[l1m0, N-1, j].item()}")
        for j in range(stage):
            print(f"  impl[{j}]  = {d.dzdt.impl[l1m0, N-1, j].item()}")

    # Implicit solve
    lm_loop()

    print(f"\n--- Stage {stage} ---")

    # Compare IC fields
    for name in ('aj_ic', 'b_ic', 'dj_ic', 'db_ic'):
        fname = f'{name}_stage{stage}'
        try:
            f_ref = load_fortran_complex(fname, DUMP_DIR)
            f_ref = reorder_snake_to_st(f_ref)
            py = getattr(fields, name).cpu().numpy()
            maxabs = max(np.abs(f_ref).max(), 1e-30)
            relerr = np.abs(py - f_ref).max() / maxabs
            print(f"  {name:12s}: rel err = {relerr:.6e}")
        except FileNotFoundError:
            print(f"  {name:12s}: no Fortran ref")

    # Compare OC fields — with per-mode breakdown for z at stage 2
    from magic_torch.blocking import st_lm2l as _st_lm2l, st_lm2m as _st_lm2m
    for name_py, name_f in [('aj_LMloc', 'aj'), ('b_LMloc', 'b'),
                             ('z_LMloc', 'z'), ('w_LMloc', 'w'), ('s_LMloc', 's')]:
        fname = f'{name_f}_stage{stage}'
        try:
            f_ref = load_fortran_complex(fname, DUMP_DIR)
            f_ref = reorder_snake_to_st(f_ref)
            py = getattr(fields, name_py).cpu().numpy()
            maxabs = max(np.abs(f_ref).max(), 1e-30)
            relerr = np.abs(py - f_ref).max() / maxabs
            print(f"  {name_py:12s}: rel err = {relerr:.6e}")
            # Show per-mode breakdown for z at stage 2
            if name_py == 'z_LMloc' and stage == 2 and relerr > 1e-10:
                abs_err = np.abs(py - f_ref)
                for lm_idx in np.argsort(abs_err.max(axis=1))[-10:][::-1]:
                    l_val = _st_lm2l[lm_idx].item()
                    m_val = _st_lm2m[lm_idx].item()
                    max_err_lm = abs_err[lm_idx].max()
                    max_nr = abs_err[lm_idx].argmax()
                    ref_val = f_ref[lm_idx, max_nr]
                    py_val = py[lm_idx, max_nr]
                    rel_lm = max_err_lm / max(np.abs(f_ref[lm_idx]).max(), 1e-30)
                    print(f"    l={l_val:2d} m={m_val:2d} lm={lm_idx:4d}: abs={max_err_lm:.4e} rel={rel_lm:.4e} at nr={max_nr} (py={py_val:.6e} ref={ref_val:.6e})")
        except FileNotFoundError:
            print(f"  {name_py:12s}: no Fortran ref")

    # Compare omega_ic
    fname_re = os.path.join(DUMP_DIR, f'omega_ic_stage{stage}.dat')
    if os.path.exists(fname_re):
        omega_ref = load_fortran_dat(fname_re).item()
        omega_py = fields.omega_ic
        relerr = abs(omega_py - omega_ref) / abs(omega_ref)
        print(f"  {'omega_ic':12s}: rel err = {relerr:.6e} (py={omega_py:.10e}, ref={omega_ref:.10e})")

    tscheme.istage = stage + 1

print("\nDone!")
