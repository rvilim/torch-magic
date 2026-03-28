"""Phase 1: Verify radial infrastructure against Fortran reference."""

import torch
from conftest import load_ref
from magic_torch.chebyshev import r, rMat, drMat, d2rMat, d3rMat
from magic_torch.radial_functions import or1, or2, rgrav, kappa, visc, lambda_, temp0, orho1, beta


def test_r():
    ref = load_ref("r")
    torch.testing.assert_close(r, ref, atol=0, rtol=1e-14)


def test_or1():
    ref = load_ref("or1")
    torch.testing.assert_close(or1, ref, atol=0, rtol=1e-14)


def test_or2():
    ref = load_ref("or2")
    torch.testing.assert_close(or2, ref, atol=0, rtol=1e-14)


def test_rgrav():
    ref = load_ref("rgrav")
    torch.testing.assert_close(rgrav, ref, atol=0, rtol=1e-14)


def test_rMat():
    ref = load_ref("rMat")
    # 32-step Chebyshev recursion: FP accumulation at machine epsilon level
    # max_abs=2.3e-14, max_val=1.0 → relative ~2.3e-14
    torch.testing.assert_close(rMat, ref, atol=1e-13, rtol=1e-13)


def test_drMat():
    ref = load_ref("drMat")
    # drMat: max_abs=2.6e-11, max_val=2048 → relative ~1.3e-14
    # Near-zero values have higher relative error, use atol
    torch.testing.assert_close(drMat, ref, atol=1e-10, rtol=1e-13)


def test_d2rMat():
    ref = load_ref("d2rMat")
    # d2rMat: max_abs=1.5e-8, max_val=1.4e6 → relative ~1e-14
    torch.testing.assert_close(d2rMat, ref, atol=1e-7, rtol=1e-13)


def test_d3rMat():
    ref = load_ref("d3rMat")
    # d3rMat: max_abs=5.8e-6, max_val=5.7e8 → relative ~1e-14
    torch.testing.assert_close(d3rMat, ref, atol=1e-5, rtol=1e-13)


def test_kappa():
    ref = load_ref("kappa")
    torch.testing.assert_close(kappa, ref, atol=0, rtol=1e-14)


def test_visc():
    ref = load_ref("visc")
    torch.testing.assert_close(visc, ref, atol=0, rtol=1e-14)


def test_lambda():
    ref = load_ref("lambda")
    torch.testing.assert_close(lambda_, ref, atol=0, rtol=1e-14)


def test_temp0():
    ref = load_ref("temp0")
    torch.testing.assert_close(temp0, ref, atol=0, rtol=1e-14)


def test_orho1():
    ref = load_ref("orho1")
    torch.testing.assert_close(orho1, ref, atol=0, rtol=1e-14)


def test_beta():
    ref = load_ref("beta")
    torch.testing.assert_close(beta, ref, atol=0, rtol=1e-14)


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"  {name} passed")
    print("Phase 1: All radial tests passed!")
