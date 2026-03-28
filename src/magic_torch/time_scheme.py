"""CNAB2 time scheme matching multistep_schemes.f90.

Implements Crank-Nicolson / Adams-Bashforth 2 (IMEX) time stepping.
- Implicit: Crank-Nicolson with parameter alpha
- Explicit: Adams-Bashforth 2nd order

CNAB2 parameters: nold=1, nexp=2, nimp=1
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import alpha


class CNAB2:
    """CNAB2 IMEX time scheme."""

    def __init__(self):
        self.nold = 1
        self.nexp = 2
        self.nimp = 1
        self.istage = 0  # single-stage scheme

        # Weight arrays (allocated with max sizes)
        self.wimp = torch.zeros(self.nold, dtype=DTYPE, device=DEVICE)
        self.wimp_lin = torch.zeros(self.nimp + 1, dtype=DTYPE, device=DEVICE)
        self.wexp = torch.zeros(self.nexp, dtype=DTYPE, device=DEVICE)

        # Time step history: dt[0] = current, dt[1] = previous
        self.dt = torch.zeros(2, dtype=DTYPE, device=DEVICE)

    def set_weights(self):
        """Compute CNAB2 weights from current dt array.

        Matches set_weights in multistep_schemes.f90 for CNAB2 case.
        """
        dt1 = self.dt[0].item()
        dt2 = self.dt[1].item()

        # Implicit (Crank-Nicolson)
        self.wimp[0] = 1.0                     # coefficient on old state
        self.wimp_lin[0] = alpha * dt1          # weight on NEW implicit term (LHS)
        self.wimp_lin[1] = (1.0 - alpha) * dt1  # weight on OLD implicit term

        # Explicit (Adams-Bashforth 2)
        self.wexp[0] = (1.0 + 0.5 * dt1 / dt2) * dt1
        self.wexp[1] = -0.5 * dt1 * dt1 / dt2

    def start_with_ab1(self):
        """Use first-order Euler for first time step (no history available).

        Matches start_with_ab1 in multistep_schemes.f90.
        """
        self.wexp[0] = self.dt[0]
        self.wexp[1] = 0.0

    def set_imex_rhs(self, dfdt) -> torch.Tensor:
        """Assemble RHS of IMEX scheme.

        rhs = wimp[0]*old[:,:,0] + wimp_lin[1]*impl[:,:,0]
            + wexp[0]*expl[:,:,0] + wexp[1]*expl[:,:,1]

        Args:
            dfdt: TimeArray instance

        Returns:
            rhs: (lm_max, n_r_max) complex tensor
        """
        rhs = self.wimp[0] * dfdt.old[:, :, 0]

        for n_o in range(self.nimp):
            rhs = rhs + self.wimp_lin[n_o + 1] * dfdt.impl[:, :, n_o]

        for n_o in range(self.nexp):
            rhs = rhs + self.wexp[n_o] * dfdt.expl[:, :, n_o]

        return rhs

    def set_imex_rhs_scalar(self, dfdt_scalar) -> float:
        """Assemble scalar RHS of IMEX scheme.

        Args:
            dfdt_scalar: TimeScalar instance

        Returns:
            rhs: scalar value
        """
        rhs = self.wimp[0].item() * dfdt_scalar.old[0]

        for n_o in range(self.nimp):
            rhs += self.wimp_lin[n_o + 1].item() * dfdt_scalar.impl[n_o]

        for n_o in range(self.nexp):
            rhs += self.wexp[n_o].item() * dfdt_scalar.expl[n_o]

        return rhs

    def rotate_imex(self, dfdt):
        """Roll time arrays: shift history back by one slot.

        expl[:,:,1] <- expl[:,:,0], old[:,:,1] <- old[:,:,0], etc.
        For CNAB2: nexp=2 so expl has 2 slots, nold=1 so old has 1 slot (no shift),
        nimp=1 so impl has 1 slot (no shift).
        """
        # Shift explicit: slot 1 <- slot 0
        for n_o in range(self.nexp - 1, 0, -1):
            dfdt.expl[:, :, n_o] = dfdt.expl[:, :, n_o - 1].clone()

        # Shift old (nold=1 for CNAB2, so no shift needed)
        for n_o in range(self.nold - 1, 0, -1):
            dfdt.old[:, :, n_o] = dfdt.old[:, :, n_o - 1].clone()

        # Shift implicit (nimp=1 for CNAB2, so no shift needed)
        for n_o in range(self.nimp - 1, 0, -1):
            dfdt.impl[:, :, n_o] = dfdt.impl[:, :, n_o - 1].clone()

    def rotate_imex_scalar(self, dfdt_scalar):
        """Roll scalar time arrays."""
        for n_o in range(self.nexp - 1, 0, -1):
            dfdt_scalar.expl[n_o] = dfdt_scalar.expl[n_o - 1]

        for n_o in range(self.nold - 1, 0, -1):
            dfdt_scalar.old[n_o] = dfdt_scalar.old[n_o - 1]

        for n_o in range(self.nimp - 1, 0, -1):
            dfdt_scalar.impl[n_o] = dfdt_scalar.impl[n_o - 1]


# Module-level singleton
tscheme = CNAB2()
