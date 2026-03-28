"""Verify fields after N full time steps against Fortran reference.

Runs initialize_fields() + setup_initial_state() + N x one_step(), saving
snapshots of all 14 field arrays at each step. Then compares each snapshot
against Fortran dumps from step_time.f90.

This validates multi-step integration: AB2 explicit history, dt rotation,
rotate_imex, and the full nonlinear → implicit solve chain over 10 steps.
"""

import pytest
import torch
from conftest import load_ref
from magic_torch import fields
from magic_torch.params import dtmax

N_STEPS = 100

# Field attribute names
_FIELDS = ["w_LMloc", "dw_LMloc", "ddw_LMloc",
           "z_LMloc", "dz_LMloc",
           "s_LMloc", "ds_LMloc",
           "p_LMloc", "dp_LMloc",
           "b_LMloc", "db_LMloc", "ddb_LMloc",
           "aj_LMloc", "dj_LMloc"]

# snapshots[step][attr] = tensor clone
_snapshots = {}
_ran = False


def _run_all():
    global _ran
    if _ran:
        return
    _ran = True

    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import setup_initial_state, one_step, initialize_dt

    initialize_fields()
    setup_initial_state()
    initialize_dt(dtmax)

    for step in range(1, N_STEPS + 1):
        one_step(n_time_step=step, dt=dtmax)
        _snapshots[step] = {attr: getattr(fields, attr).clone() for attr in _FIELDS}


# Map ref prefix to (fields attribute, atol, rtol)
_FIELD_SPECS = {
    "w": ("w_LMloc", 1e-11, 1e-12),
    "z": ("z_LMloc", 1e-11, 1e-12),
    "s": ("s_LMloc", 1e-11, 1e-12),
    "p": ("p_LMloc", 1e-7, 1e-12),
    "b": ("b_LMloc", 1e-11, 1e-12),
    "aj": ("aj_LMloc", 1e-11, 1e-12),
    "dw": ("dw_LMloc", 1e-11, 1e-10),
    "dz": ("dz_LMloc", 1e-9, 1e-10),
    "ds": ("ds_LMloc", 1e-11, 1e-10),
    "dp": ("dp_LMloc", 1e-4, 1e-10),
    "db": ("db_LMloc", 1e-11, 1e-10),
    "dj": ("dj_LMloc", 1e-11, 1e-10),
    "ddw": ("ddw_LMloc", 1e-8, 1e-10),
    "ddb": ("ddb_LMloc", 1e-8, 1e-10),
}


@pytest.mark.parametrize("step", range(1, N_STEPS + 1))
@pytest.mark.parametrize("field_name", list(_FIELD_SPECS.keys()))
def test_field_step(field_name, step):
    _run_all()
    attr, atol, rtol = _FIELD_SPECS[field_name]
    ref = load_ref(f"{field_name}_step{step}")
    actual = _snapshots[step][attr]
    torch.testing.assert_close(actual, ref, atol=atol, rtol=rtol)
