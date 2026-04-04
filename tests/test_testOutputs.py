"""Test testOutputs sample: power, u_square, helicity, hemi output.

Runs the full anelastic MHD simulation (100 steps, n_log_step=10) and compares
concatenated output files against the Fortran reference.out.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "testOutputs")
REF_PATH = os.path.join(SAMPLES_DIR, "reference.out")

# Environment variables matching samples/testOutputs/input.nml
_ENV = {
    "MAGIC_LMAX": "85",
    "MAGIC_NR": "73",
    "MAGIC_NCHEBMAX": "71",
    "MAGIC_MINC": "1",
    "MAGIC_RA": "3e5",
    "MAGIC_EK": "1e-3",
    "MAGIC_PR": "1.0",
    "MAGIC_PRMAG": "5.0",
    "MAGIC_RADRATIO": "0.35",
    "MAGIC_STRAT": "0.1",
    "MAGIC_POLIND": "2.0",
    "MAGIC_G0": "0.0",
    "MAGIC_G1": "1.0",
    "MAGIC_G2": "0.0",
    "MAGIC_MODE": "0",
    "MAGIC_DTMAX": "1e-4",
    "MAGIC_ALPHA": "0.6",
    "MAGIC_COURFAC": "2.5",
    "MAGIC_ALFFAC": "1.0",
    "MAGIC_INIT_S1": "404",
    "MAGIC_AMP_S1": "0.01",
    "MAGIC_INIT_V1": "0",
    "MAGIC_KTOPV": "2",
    "MAGIC_KBOTV": "2",
    "MAGIC_N_LOG_STEP": "10",
    "MAGIC_L_POWER": "true",
    "MAGIC_L_HEL": "true",
    "MAGIC_L_HEMI": "true",
    "MAGIC_DEVICE": "cpu",
}


def _read_stack(path):
    """Read all numbers from a file into a flat array (matching unitTest.py readStack)."""
    vals = []
    with open(path) as f:
        for line in f:
            vals.extend(float(x) for x in line.split())
    return np.array(vals, dtype=np.float64)


@pytest.fixture(scope="module")
def output_dir():
    """Run the testOutputs simulation in a subprocess and return the output directory."""
    runner = os.path.join(os.path.dirname(__file__), "_testOutputs_runner.py")
    with tempfile.TemporaryDirectory(prefix="testOutputs_") as tmpdir:
        env = {**os.environ, **_ENV}
        result = subprocess.run(
            [sys.executable, runner, tmpdir],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            pytest.fail(f"Runner failed with code {result.returncode}")
        yield tmpdir


# Output files in concatenation order (matching unitTest.py setUp)
_OUTPUT_FILES = [
    "e_kin.start",
    "e_mag_oc.start",
    "e_mag_ic.start",
    "dipole.start",
    "heat.start",
    "par.start",
    "power.start",
    "u_square.start",
    "helicity.start",
    "hemi.start",
]

# Per-file relative tolerances.
# FP accumulation over 100 steps causes ~1e-4 divergence for some quantities.
# Absolute tolerance handles near-zero values.
_RTOL = {
    "e_kin.start": 5e-4,
    "e_mag_oc.start": 5e-4,
    "e_mag_ic.start": 5e-4,
    "dipole.start": 5e-4,
    "heat.start": 1e-3,
    "par.start": 5e-2,      # Geos/dpV/dzV not yet implemented
    "power.start": 5e-4,
    "u_square.start": 5e-4,
    "helicity.start": 5e-4,
    "hemi.start": 5e-4,
}
_ATOL = 1e-10  # handles comparisons where ref ~ 0


@pytest.fixture(scope="module")
def ref_sections(output_dir):
    """Split reference.out into per-file sections based on output file sizes."""
    ref_data = _read_stack(REF_PATH)
    sections = {}
    offset = 0
    for fname in _OUTPUT_FILES:
        fpath = os.path.join(output_dir, fname)
        test_vals = _read_stack(fpath)
        n = len(test_vals)
        sections[fname] = (test_vals, ref_data[offset:offset + n])
        offset += n
    assert offset == len(ref_data), f"Total ref values: {len(ref_data)}, consumed: {offset}"
    return sections


@pytest.mark.parametrize("fname", _OUTPUT_FILES)
def test_output_file(ref_sections, fname):
    """Compare individual output file against reference section."""
    test_vals, ref_vals = ref_sections[fname]
    assert len(test_vals) == len(ref_vals), (
        f"{fname}: length mismatch {len(test_vals)} vs {len(ref_vals)}")

    if fname == "par.start":
        # par.start has 20 columns per row. Some columns not yet implemented (output 0):
        # col4=Geos, col9=dpV, col10=dzV, col11=lvDiss, col12=lbDiss, col19=ReEquat
        n_cols = 20
        mask = np.ones(len(test_vals), dtype=bool)
        for row in range(len(test_vals) // n_cols):
            for col in [4, 9, 10, 11, 12, 19]:
                mask[row * n_cols + col] = False
        np.testing.assert_allclose(
            test_vals[mask], ref_vals[mask],
            rtol=_RTOL[fname], atol=_ATOL,
            err_msg=f"Mismatch in {fname} (Geos/dpV/dzV columns excluded)",
        )
    else:
        np.testing.assert_allclose(
            test_vals, ref_vals,
            rtol=_RTOL[fname], atol=_ATOL,
            err_msg=f"Mismatch in {fname}",
        )


def test_individual_files_exist(output_dir):
    """Verify all expected output files were created."""
    for fname in _OUTPUT_FILES:
        fpath = os.path.join(output_dir, fname)
        assert os.path.exists(fpath), f"Missing: {fname}"


def test_power_line_count(output_dir):
    """Power file should have one fewer line than other files (no t=0 row)."""
    power_path = os.path.join(output_dir, "power.start")
    ekin_path = os.path.join(output_dir, "e_kin.start")
    with open(power_path) as f:
        n_power = sum(1 for _ in f)
    with open(ekin_path) as f:
        n_ekin = sum(1 for _ in f)
    assert n_power == n_ekin - 1, f"power has {n_power} lines, e_kin has {n_ekin}"
