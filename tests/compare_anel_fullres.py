"""Compare full-resolution anelastic energy against Fortran reference.out."""
import csv
import numpy as np
from pathlib import Path

REF_FILE = Path(__file__).parent.parent.parent / "samples" / "hydro_bench_anel" / "reference.out"
LOG_FILE = Path("/tmp/anel_fullres/log.csv")


def load_reference():
    """Load Fortran reference.out — columns: time, e_kin_pol, e_kin_tor, ..."""
    data = np.loadtxt(REF_FILE)
    return data[:, 0], data[:, 1], data[:, 2]  # time, e_kin_pol, e_kin_tor


def load_python_log():
    """Load Python log.csv and extract energy at matching times."""
    times, pols, tors = [], [], []
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["sim_time"]))
            pols.append(float(row["e_kin_pol"]))
            tors.append(float(row["e_kin_tor"]))
    return np.array(times), np.array(pols), np.array(tors)


def main():
    ref_t, ref_pol, ref_tor = load_reference()
    py_t, py_pol, py_tor = load_python_log()

    print(f"Reference: {len(ref_t)} time points (t={ref_t[0]:.4e} to {ref_t[-1]:.4e})")
    print(f"Python:    {len(py_t)} time points (t={py_t[0]:.4e} to {py_t[-1]:.4e})")
    print()

    # Match times (reference is every 1e-3, python is every 10*1e-4 = 1e-3)
    max_pol_err = 0.0
    max_tor_err = 0.0
    print(f"{'time':>12s} {'ref_pol':>14s} {'py_pol':>14s} {'rel_err_pol':>12s} {'ref_tor':>14s} {'py_tor':>14s} {'rel_err_tor':>12s}")
    print("-" * 95)

    for i, t_ref in enumerate(ref_t):
        # Find matching python time
        idx = np.argmin(np.abs(py_t - t_ref))
        if abs(py_t[idx] - t_ref) > 1e-8:
            continue

        rp = ref_pol[i]
        pp = py_pol[idx]
        rt = ref_tor[i]
        pt = py_tor[idx]

        err_pol = abs(pp - rp) / max(abs(rp), 1e-30)
        err_tor = abs(pt - rt) / max(abs(rt), 1e-30)

        max_pol_err = max(max_pol_err, err_pol)
        max_tor_err = max(max_tor_err, err_tor)

        if i % 5 == 0 or i == len(ref_t) - 1:
            print(f"{t_ref:12.4e} {rp:14.8e} {pp:14.8e} {err_pol:12.3e} {rt:14.8e} {pt:14.8e} {err_tor:12.3e}")

    print("-" * 95)
    print(f"Max relative errors: e_kin_pol={max_pol_err:.3e}, e_kin_tor={max_tor_err:.3e}")


if __name__ == "__main__":
    main()
