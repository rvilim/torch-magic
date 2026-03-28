"""Type-I Discrete Cosine Transform via torch.fft.

Implements a self-inverse (involutory) DCT-I matching the Fortran costf1
from cosine_transform_odd.f90. The transform is its own inverse:
costf(costf(f)) = f.

The implementation uses the standard trick of reducing DCT-I to FFT via
symmetric extension. The unnormalized DCT-I matrix M satisfies M^2 = 2N*I,
so the self-inverse transform is S = M / sqrt(2N).
"""

import math
import torch

from .precision import DTYPE, CDTYPE, DEVICE


def costf(f: torch.Tensor) -> torch.Tensor:
    """Self-inverse Type-I DCT.

    For input of shape (..., n_r_max), applies the DCT-I along the last
    dimension. Works for both real and complex tensors.

    The unnormalized DCT-I (M) computes:
        M(f)[k] = f[0] + (-1)^k*f[N] + 2*sum_{j=1}^{N-1} f[j]*cos(j*k*pi/N)
    Since M^2 = 2N*I, we normalize by 1/sqrt(2N) to get a self-inverse transform.
    """
    n = f.shape[-1]  # n_r_max
    N = n - 1        # Chebyshev degree (= n_r_max - 1)

    # Symmetric extension: [f[0], f[1], ..., f[N], f[N-1], ..., f[1]]  (length 2N)
    f_ext = torch.cat([f, f[..., 1:-1].flip(-1)], dim=-1)

    # FFT of the symmetric extension gives the unnormalized DCT-I
    F = torch.fft.fft(f_ext, dim=-1)
    Y = F[..., :n]

    # For real input, imaginary part is numerical noise
    if not f.is_complex():
        Y = Y.real

    # Normalize to make self-inverse: S = M / sqrt(2N)
    Y = Y / math.sqrt(2.0 * N)

    return Y
