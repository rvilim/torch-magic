"""Phase 0: Verify blocking/LM mappings against Fortran reference."""

import torch
from conftest import load_ref
from magic_torch.blocking import st_lm2l, st_lm2m, st_lm2lmS, st_lm2lmA


def test_lm2l():
    ref = load_ref("lm2l")
    # Fortran is 1-indexed, PyTorch is 0-indexed
    # The actual l values should be the same
    torch.testing.assert_close(st_lm2l, ref.long(), atol=0, rtol=0)


def test_lm2m():
    ref = load_ref("lm2m")
    torch.testing.assert_close(st_lm2m, ref.long(), atol=0, rtol=0)


def test_lm2lmS():
    ref = load_ref("lm2lmS")
    # Fortran uses 1-based indexing, so lm2lmS values need adjustment
    # In Fortran, lm2lmS(lm) points to another lm index (1-based)
    # In PyTorch, lm2lmS[lm] is 0-based
    # Fortran: for l==m (no symmetric partner), lm2lmS = lm (self-reference)
    # PyTorch: for l==m, lm2lmS = lm (also self-reference)
    # Adjust: Fortran 1-based -> 0-based: ref - 1
    ref_0based = ref.long() - 1  # Convert 1-based to 0-based
    torch.testing.assert_close(st_lm2lmS, ref_0based, atol=0, rtol=0)


def test_lm2lmA():
    ref = load_ref("lm2lmA")
    # Fortran: lm2lmA = -1 for l==l_max (no antisymmetric partner)
    # PyTorch: lm2lmA = -1 for l==l_max
    # For valid entries, Fortran is 1-based -> convert to 0-based
    ref_0based = ref.long().clone()
    valid = ref_0based > 0  # Fortran uses -1 for invalid, >0 for valid (1-based)
    ref_0based[valid] -= 1  # Convert valid entries to 0-based
    torch.testing.assert_close(st_lm2lmA, ref_0based, atol=0, rtol=0)


if __name__ == "__main__":
    test_lm2l()
    test_lm2m()
    test_lm2lmS()
    test_lm2lmA()
    print("Phase 0: All blocking tests passed!")
