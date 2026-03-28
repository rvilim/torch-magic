"""Compare Fortran vs Python energies over 1000 steps.

Parses Fortran e_kin.test and e_mag_oc.test, runs Python for 1000 steps,
and outputs a CSV comparison + summary statistics.
"""

import csv
import sys
import time as timeit
from pathlib import Path

import numpy as np


def parse_fortran_energies(samples_dir: Path):
    """Parse Fortran energy output files.

    e_kin.test columns: time, e_kin_pol, e_kin_tor, ...
    e_mag_oc.test columns: time, e_mag_pol, e_mag_tor, ...

    Returns:
        times, ekin_pol, ekin_tor, emag_pol, emag_tor: numpy arrays
    """
    ekin_data = np.loadtxt(samples_dir / "e_kin.test")
    emag_data = np.loadtxt(samples_dir / "e_mag_oc.test")

    times = ekin_data[:, 0]
    ekin_pol = ekin_data[:, 1]
    ekin_tor = ekin_data[:, 2]
    emag_pol = emag_data[:, 1]
    emag_tor = emag_data[:, 2]

    return times, ekin_pol, ekin_tor, emag_pol, emag_tor


def run_python_energies(n_steps: int):
    """Run Python simulation and collect energies at each step.

    Returns:
        list of (ekin_pol, ekin_tor, emag_pol, emag_tor) tuples, length n_steps+1
    """
    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import setup_initial_state, one_step, initialize_dt
    from magic_torch.output import get_e_kin, get_e_mag
    from magic_torch import fields
    from magic_torch.params import dtmax

    dt = dtmax

    initialize_fields()
    setup_initial_state()
    initialize_dt(dt)

    energies = []

    # Step 0: initial state
    ekp, ekt = get_e_kin(
        fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
        fields.z_LMloc, fields.dz_LMloc)
    emp, emt = get_e_mag(
        fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
        fields.aj_LMloc, fields.dj_LMloc)
    energies.append((ekp, ekt, emp, emt))

    t_start = timeit.time()
    for n in range(1, n_steps + 1):
        one_step(n, dt)

        ekp, ekt = get_e_kin(
            fields.w_LMloc, fields.dw_LMloc, fields.ddw_LMloc,
            fields.z_LMloc, fields.dz_LMloc)
        emp, emt = get_e_mag(
            fields.b_LMloc, fields.db_LMloc, fields.ddb_LMloc,
            fields.aj_LMloc, fields.dj_LMloc)
        energies.append((ekp, ekt, emp, emt))

        elapsed = timeit.time() - t_start
        print(f"  Step {n:5d}/{n_steps}: "
              f"e_kin={ekp+ekt:.6e}, e_mag={emp+emt:.6e}, "
              f"wall={elapsed:.1f}s")

    elapsed = timeit.time() - t_start
    print(f"  Done: {n_steps} steps in {elapsed:.1f}s "
          f"({elapsed/n_steps:.3f}s/step)")

    return energies, dt


def main():
    n_steps = 1000

    # Paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    samples_dir = repo_root / "samples" / "dynamo_benchmark"
    output_csv = Path(__file__).resolve().parent.parent / "energy_comparison.csv"

    # Parse Fortran
    print("Parsing Fortran energy files...")
    f_times, f_ekp, f_ekt, f_emp, f_emt = parse_fortran_energies(samples_dir)
    n_fortran = len(f_times) - 1  # exclude header/step-0 depends on file
    print(f"  Fortran: {len(f_times)} rows (steps 0..{len(f_times)-1})")

    if len(f_times) < n_steps + 1:
        print(f"WARNING: Fortran only has {len(f_times)} rows, "
              f"reducing to {len(f_times)-1} steps")
        n_steps = len(f_times) - 1

    # Run Python
    print(f"\nRunning Python for {n_steps} steps...")
    py_energies, dt = run_python_energies(n_steps)

    # Write CSV
    print(f"\nWriting {output_csv}...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "time",
            "fortran_ekin_pol", "fortran_ekin_tor",
            "python_ekin_pol", "python_ekin_tor",
            "fortran_emag_pol", "fortran_emag_tor",
            "python_emag_pol", "python_emag_tor",
        ])
        for i in range(n_steps + 1):
            sim_time = f_times[i]
            p_ekp, p_ekt, p_emp, p_emt = py_energies[i]
            writer.writerow([
                i, f"{sim_time:.12e}",
                f"{f_ekp[i]:.8e}", f"{f_ekt[i]:.8e}",
                f"{p_ekp:.8e}", f"{p_ekt:.8e}",
                f"{f_emp[i]:.8e}", f"{f_emt[i]:.8e}",
                f"{p_emp:.8e}", f"{p_emt:.8e}",
            ])

    # Summary statistics
    print("\n=== Energy Comparison Summary ===")
    print(f"{'Component':<15} {'Max Rel Err':>12} {'Max Abs Err':>12} "
          f"{'Worst Step':>10}")

    labels = ["ekin_pol", "ekin_tor", "emag_pol", "emag_tor"]
    fortran_arrays = [f_ekp, f_ekt, f_emp, f_emt]

    for idx, (label, f_arr) in enumerate(zip(labels, fortran_arrays)):
        py_arr = np.array([e[idx] for e in py_energies])
        f_vals = f_arr[:n_steps + 1]

        abs_err = np.abs(py_arr - f_vals)
        # Relative error, avoiding division by zero
        denom = np.maximum(np.abs(f_vals), 1e-30)
        rel_err = abs_err / denom

        worst = np.argmax(rel_err)
        print(f"  {label:<13} {rel_err[worst]:12.4e} {abs_err[worst]:12.4e} "
              f"{worst:10d}")

    # Also print at a few milestones
    print("\n=== Milestones ===")
    milestones = [0, 1, 10, 100, 500, 1000]
    for step in milestones:
        if step > n_steps:
            break
        p_ekp, p_ekt, p_emp, p_emt = py_energies[step]
        print(f"Step {step:5d}: "
              f"ekin_pol rel={abs(p_ekp - f_ekp[step])/max(abs(f_ekp[step]), 1e-30):.2e}, "
              f"ekin_tor rel={abs(p_ekt - f_ekt[step])/max(abs(f_ekt[step]), 1e-30):.2e}, "
              f"emag_pol rel={abs(p_emp - f_emp[step])/max(abs(f_emp[step]), 1e-30):.2e}, "
              f"emag_tor rel={abs(p_emt - f_emt[step])/max(abs(f_emt[step]), 1e-30):.2e}")


if __name__ == "__main__":
    main()
