"""Phase 2: Verify horizontal data against Fortran reference.

Note: Fortran dumps gauss and O_sin_theta_E2 in interleaved N/S order
(l_scramble_theta=.true.), while PyTorch stores sorted and _grid variants.
"""

import torch
from conftest import load_ref
from magic_torch.horizontal_data import (
    gauss, theta_ord, dLh,
    dTheta1S, dTheta1A, dTheta2S, dTheta2A,
    dTheta3S, dTheta3A, dTheta4S, dTheta4A,
    O_sin_theta_E2_grid, _grid_idx,
)


def test_gauss():
    """Fortran gauss is in interleaved order; compare sorted values."""
    ref = load_ref("gauss")
    # Fortran gauss is interleaved N/S, reorder to sorted for comparison
    gauss_interleaved = gauss[_grid_idx]
    torch.testing.assert_close(gauss_interleaved, ref, atol=1e-15, rtol=1e-14)


def test_theta_ord():
    ref = load_ref("theta_ord")
    # theta_ord is in sorted order in both Fortran and PyTorch
    torch.testing.assert_close(theta_ord, ref, atol=1e-15, rtol=1e-14)


def test_dLh():
    ref = load_ref("dLh")
    torch.testing.assert_close(dLh, ref, atol=0, rtol=1e-14)


def test_dTheta1S():
    ref = load_ref("dTheta1S")
    torch.testing.assert_close(dTheta1S, ref, atol=0, rtol=1e-14)


def test_dTheta1A():
    ref = load_ref("dTheta1A")
    torch.testing.assert_close(dTheta1A, ref, atol=0, rtol=1e-14)


def test_dTheta2S():
    ref = load_ref("dTheta2S")
    torch.testing.assert_close(dTheta2S, ref, atol=0, rtol=1e-14)


def test_dTheta2A():
    ref = load_ref("dTheta2A")
    torch.testing.assert_close(dTheta2A, ref, atol=0, rtol=1e-14)


def test_dTheta3S():
    ref = load_ref("dTheta3S")
    torch.testing.assert_close(dTheta3S, ref, atol=0, rtol=1e-14)


def test_dTheta3A():
    ref = load_ref("dTheta3A")
    torch.testing.assert_close(dTheta3A, ref, atol=0, rtol=1e-14)


def test_dTheta4S():
    ref = load_ref("dTheta4S")
    torch.testing.assert_close(dTheta4S, ref, atol=0, rtol=1e-14)


def test_dTheta4A():
    ref = load_ref("dTheta4A")
    torch.testing.assert_close(dTheta4A, ref, atol=0, rtol=1e-14)


def test_O_sin_theta_E2():
    """Fortran O_sin_theta_E2 is in interleaved order; compare _grid variant."""
    ref = load_ref("O_sin_theta_E2")
    # FP diffs from 1/sin^2 computation: max_abs=2.4e-12 at poles where values are ~100
    torch.testing.assert_close(O_sin_theta_E2_grid, ref, atol=1e-11, rtol=1e-13)


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"  {name} passed")
    print("Phase 2: All horizontal tests passed!")
