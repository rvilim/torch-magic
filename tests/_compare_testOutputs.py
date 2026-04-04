"""Detailed comparison of testOutputs output vs reference."""
import os
import sys
import subprocess
import tempfile
import numpy as np

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "testOutputs")
REF_PATH = os.path.join(SAMPLES_DIR, "reference.out")

_ENV = {
    "MAGIC_LMAX": "85", "MAGIC_NR": "73", "MAGIC_NCHEBMAX": "71", "MAGIC_MINC": "1",
    "MAGIC_RA": "3e5", "MAGIC_EK": "1e-3", "MAGIC_PR": "1.0", "MAGIC_PRMAG": "5.0",
    "MAGIC_RADRATIO": "0.35", "MAGIC_STRAT": "0.1", "MAGIC_POLIND": "2.0",
    "MAGIC_G0": "0.0", "MAGIC_G1": "1.0", "MAGIC_G2": "0.0",
    "MAGIC_MODE": "0", "MAGIC_DTMAX": "1e-4", "MAGIC_ALPHA": "0.6",
    "MAGIC_COURFAC": "2.5", "MAGIC_ALFFAC": "1.0",
    "MAGIC_INIT_S1": "404", "MAGIC_AMP_S1": "0.01", "MAGIC_INIT_V1": "0",
    "MAGIC_KTOPV": "2", "MAGIC_KBOTV": "2",
    "MAGIC_N_LOG_STEP": "10", "MAGIC_L_POWER": "true",
    "MAGIC_L_HEL": "true", "MAGIC_L_HEMI": "true", "MAGIC_DEVICE": "cpu",
}

_OUTPUT_FILES = [
    "e_kin.start", "e_mag_oc.start", "e_mag_ic.start", "dipole.start",
    "heat.start", "par.start", "power.start", "u_square.start",
    "helicity.start", "hemi.start",
]


def _read_stack(path):
    vals = []
    with open(path) as f:
        for line in f:
            vals.extend(float(x) for x in line.split())
    return np.array(vals, dtype=np.float64)


def main():
    runner = os.path.join(os.path.dirname(__file__), "_testOutputs_runner.py")
    with tempfile.TemporaryDirectory(prefix="testOutputs_") as tmpdir:
        env = {**os.environ, **_ENV}
        result = subprocess.run(
            [sys.executable, runner, tmpdir], env=env,
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
            sys.exit(1)

        # Load reference as a flat array
        ref_data = _read_stack(REF_PATH)
        ref_offset = 0

        # Compare per-file
        for fname in _OUTPUT_FILES:
            fpath = os.path.join(tmpdir, fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fname}")
                continue
            test_vals = _read_stack(fpath)
            n = len(test_vals)
            ref_vals = ref_data[ref_offset:ref_offset + n]
            ref_offset += n

            # Compute errors
            mask = np.abs(ref_vals) > 1e-20
            if mask.any():
                rel_err = np.abs(test_vals[mask] - ref_vals[mask]) / np.abs(ref_vals[mask])
                max_rel = rel_err.max()
                max_abs = np.abs(test_vals - ref_vals).max()
                n_bad = np.sum(rel_err > 1e-8)
            else:
                max_rel = 0.0
                max_abs = np.abs(test_vals - ref_vals).max()
                n_bad = 0

            status = "OK" if max_rel < 1e-6 else "FAIL"
            print(f"  {fname:25s}: max_rel={max_rel:.2e}  max_abs={max_abs:.2e}  "
                  f"n_bad(>1e-8)={n_bad:4d}/{n:4d}  {status}")

            if max_rel > 1e-6:
                # Show first few mismatches
                for i in range(min(len(test_vals), len(ref_vals))):
                    if abs(ref_vals[i]) > 1e-20:
                        re = abs(test_vals[i] - ref_vals[i]) / abs(ref_vals[i])
                    else:
                        re = abs(test_vals[i] - ref_vals[i])
                    if re > 1e-6:
                        print(f"    [{i:4d}] test={test_vals[i]:.10e} ref={ref_vals[i]:.10e} rel={re:.2e}")
                        if i > 20:
                            print("    ... (truncated)")
                            break

        print(f"\nTotal ref values: {len(ref_data)}, consumed: {ref_offset}")


if __name__ == "__main__":
    main()
