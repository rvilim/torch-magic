"""Time schemes: CNAB2 (multistep) and BPR353 (SDIRK).

CNAB2: Crank-Nicolson / Adams-Bashforth 2 (IMEX), nold=1, nexp=2, nimp=1
BPR353: 4-stage SDIRK (Boscarino-Pareschi-Russo 2013), nold=1, nexp=4, nimp=4
"""

import torch

from .precision import DTYPE, CDTYPE, DEVICE
from .params import alpha, courfac as _courfac_nml, alffac as _alffac_nml, intfac as _intfac_nml


def _scheme_courfac(courfac_loc, alffac_loc, intfac_loc):
    """Resolve courfac/alffac/intfac: use namelist value if < 1000, else scheme default."""
    cf = courfac_loc if abs(_courfac_nml) >= 1e3 else _courfac_nml
    af = alffac_loc if abs(_alffac_nml) >= 1e3 else _alffac_nml
    intf = intfac_loc if abs(_intfac_nml) >= 1e3 else _intfac_nml
    return cf, af, intf


class CNAB2:
    """CNAB2 IMEX time scheme."""

    nstages = 1

    def __init__(self):
        self.family = "MULTISTEP"
        self.nold = 1
        self.nexp = 2
        self.nimp = 1
        self.istage = 0  # single-stage scheme
        # CFL factors (multistep_schemes.f90: CNAB2 courfac_loc=2.5, alffac_loc=1.0)
        self.courfac, self.alffac, self.intfac = _scheme_courfac(2.5, 1.0, 0.15)

        # Weight arrays (allocated with max sizes)
        self.wimp = torch.zeros(self.nold, dtype=DTYPE, device=DEVICE)
        self.wimp_lin = torch.zeros(self.nimp + 1, dtype=DTYPE, device=DEVICE)
        self.wexp = torch.zeros(self.nexp, dtype=DTYPE, device=DEVICE)

        # Time step history: dt[0] = current, dt[1] = previous
        self.dt = torch.zeros(2, dtype=DTYPE, device=DEVICE)

        # Stage control (trivial for CNAB2)
        self.l_exp_calc = [True]

    @property
    def next_impl_idx(self):
        """Index in impl array to store the next implicit term. Always 0 for CNAB2."""
        return 0

    @property
    def store_old(self):
        """Whether to store old state after this solve. Always True for CNAB2."""
        return True

    def set_weights(self):
        """Compute CNAB2 weights from current dt array."""
        dt1 = self.dt[0].item()
        dt2 = self.dt[1].item()

        # Implicit (Crank-Nicolson)
        self.wimp[0] = 1.0
        self.wimp_lin[0] = alpha * dt1
        self.wimp_lin[1] = (1.0 - alpha) * dt1

        # Explicit (Adams-Bashforth 2)
        self.wexp[0] = (1.0 + 0.5 * dt1 / dt2) * dt1
        self.wexp[1] = -0.5 * dt1 * dt1 / dt2

        # Cache as Python floats to avoid .item() GPU syncs per step
        self._wimp_py = [self.wimp[i].item() for i in range(self.nold)]
        self._wimp_lin_py = [self.wimp_lin[i].item() for i in range(self.nimp + 1)]
        self._wexp_py = [self.wexp[i].item() for i in range(self.nexp)]

    def start_with_ab1(self):
        """Use first-order Euler for first time step (no history available)."""
        self.wexp[0] = self.dt[0]
        self.wexp[1] = 0.0
        # Update cached Python floats
        self._wexp_py = [self.wexp[0].item(), 0.0]

    def set_imex_rhs(self, dfdt) -> torch.Tensor:
        """Assemble RHS of IMEX scheme.

        rhs = wimp[0]*old[:,:,0] + wimp_lin[1]*impl[:,:,0]
            + wexp[0]*expl[:,:,0] + wexp[1]*expl[:,:,1]
        """
        rhs = self._wimp_py[0] * dfdt.old[:, :, 0]

        for n_o in range(self.nimp):
            rhs = rhs + self._wimp_lin_py[n_o + 1] * dfdt.impl[:, :, n_o]

        for n_o in range(self.nexp):
            rhs = rhs + self._wexp_py[n_o] * dfdt.expl[:, :, n_o]

        return rhs

    def set_imex_rhs_scalar(self, dfdt_scalar) -> float:
        """Assemble scalar RHS of IMEX scheme."""
        rhs = self._wimp_py[0] * dfdt_scalar.old[0]

        for n_o in range(self.nimp):
            rhs += self._wimp_lin_py[n_o + 1] * dfdt_scalar.impl[n_o]

        for n_o in range(self.nexp):
            rhs += self._wexp_py[n_o] * dfdt_scalar.expl[n_o]

        return rhs

    def set_imex_rhs_multi(self, dfdts) -> torch.Tensor:
        """Assemble stacked RHS for multiple dfdts at once.

        Reduces MPS dispatch overhead by operating on larger tensors.
        Returns (n * lm_max, n_r_max) with results stacked vertically.
        """
        rhs = torch.cat([self._wimp_py[0] * dt.old[:, :, 0] for dt in dfdts], dim=0)

        for n_o in range(self.nimp):
            w = self._wimp_lin_py[n_o + 1]
            if w != 0.0:
                rhs = rhs + w * torch.cat([dt.impl[:, :, n_o] for dt in dfdts], dim=0)

        for n_o in range(self.nexp):
            w = self._wexp_py[n_o]
            if w != 0.0:
                rhs = rhs + w * torch.cat([dt.expl[:, :, n_o] for dt in dfdts], dim=0)

        return rhs

    def rotate_imex(self, dfdt):
        """Roll time arrays: shift history back by one slot.

        No .clone() needed: reverse iteration order ensures each source is read
        before being overwritten, and src/dst are non-overlapping dim-2 slices.
        """
        for n_o in range(self.nexp - 1, 0, -1):
            dfdt.expl[:, :, n_o] = dfdt.expl[:, :, n_o - 1]

        for n_o in range(self.nold - 1, 0, -1):
            dfdt.old[:, :, n_o] = dfdt.old[:, :, n_o - 1]

        for n_o in range(self.nimp - 1, 0, -1):
            dfdt.impl[:, :, n_o] = dfdt.impl[:, :, n_o - 1]

    def rotate_imex_scalar(self, dfdt_scalar):
        """Roll scalar time arrays."""
        for n_o in range(self.nexp - 1, 0, -1):
            dfdt_scalar.expl[n_o] = dfdt_scalar.expl[n_o - 1]

        for n_o in range(self.nold - 1, 0, -1):
            dfdt_scalar.old[n_o] = dfdt_scalar.old[n_o - 1]

        for n_o in range(self.nimp - 1, 0, -1):
            dfdt_scalar.impl[n_o] = dfdt_scalar.impl[n_o - 1]


class BPR353:
    """BPR353 4-stage SDIRK IMEX time scheme.

    Boscarino-Pareschi-Russo (2013), matching dirk_schemes.f90.

    The impl/expl arrays use a "store one ahead" convention:
    - After solving stage k (1-based), impl is stored at index k+1 (wrapping to 1)
    - impl[j] holds L*u_{j-1} (one stage behind)
    - old[1] is set only at the wrap point (after last stage)

    In 0-based Python indexing:
    - impl[:,:,j] holds L*u_{stage j-1} where u_0 = u_n
    - After solving stage k (istage=k, 1-based), store impl at (k % nstages) (0-based)
    - old[:,:,0] is set only when istage == nstages
    """

    nstages = 4

    def __init__(self):
        self.family = "DIRK"
        self.nold = 1
        self.nexp = 4
        self.nimp = 4
        self.istage = 1  # 1-based, reset at start of each time step
        # CFL factors (dirk_schemes.f90: BPR353 courfac_loc=0.8, alffac_loc=0.35)
        self.courfac, self.alffac, self.intfac = _scheme_courfac(0.8, 0.35, 0.46)

        # Butcher tableau (5x5), unscaled — will be multiplied by dt in set_weights
        # Row 0 is all zeros (initial state, unused)
        # Row k (1-4) used for stage k
        self._butcher_imp_raw = torch.tensor([
            [0.0,    0.0,   0.0,   0.0,   0.0],
            [1/2,    1/2,   0.0,   0.0,   0.0],
            [5/18,  -1/9,   1/2,   0.0,   0.0],
            [1/2,    0.0,   0.0,   1/2,   0.0],
            [1/4,    0.0,   3/4,  -1/2,   1/2],
        ], dtype=DTYPE)

        self._butcher_exp_raw = torch.tensor([
            [0.0,    0.0,   0.0,   0.0,   0.0],
            [1.0,    0.0,   0.0,   0.0,   0.0],
            [4/9,    2/9,   0.0,   0.0,   0.0],
            [1/4,    0.0,   3/4,   0.0,   0.0],
            [1/4,    0.0,   3/4,   0.0,   0.0],
        ], dtype=DTYPE)

        # Stage times (c vector): t_stage = t_n + dt * c[k-1] for stage k
        self.butcher_c = torch.tensor([1.0, 2/3, 1.0, 1.0], dtype=DTYPE)

        # Which stages compute explicit terms / implicit RHS
        self.l_exp_calc = [True, True, True, False]
        self.l_imp_calc_rhs = [True, True, True, True]

        # wimp_lin: [0] = SDIRK diagonal * dt (set in set_weights)
        self.wimp_lin = torch.zeros(1, dtype=DTYPE, device=DEVICE)
        self._wimp_lin_raw = 0.5  # SDIRK diagonal (constant for all stages)

        # Scaled Butcher tables (set in set_weights)
        self.butcher_imp = torch.zeros(5, 5, dtype=DTYPE, device=DEVICE)
        self.butcher_exp = torch.zeros(5, 5, dtype=DTYPE, device=DEVICE)

        # Time step: single value for DIRK
        self.dt = torch.zeros(1, dtype=DTYPE, device=DEVICE)

    @property
    def next_impl_idx(self):
        """0-based index for storing impl after current stage solve.

        After solving stage k (1-based), store at (k % nstages).
        """
        return self.istage % self.nstages

    @property
    def store_old(self):
        """Whether to store old state after current solve.

        Only at the wrap point: after last stage, impl wraps to index 0.
        """
        return self.istage == self.nstages

    def set_weights(self):
        """Scale Butcher tableau by dt and cache as Python floats."""
        dt_val = self.dt[0].item()
        self.wimp_lin[0] = self._wimp_lin_raw * dt_val
        self.butcher_imp.copy_(self._butcher_imp_raw * dt_val)
        self.butcher_exp.copy_(self._butcher_exp_raw * dt_val)
        # Cache as Python float lists to avoid .item() GPU syncs per step
        self._imp_py = self.butcher_imp.tolist()
        self._exp_py = self.butcher_exp.tolist()

    def start_with_ab1(self):
        """No-op for DIRK schemes (no history needed)."""
        pass

    def set_imex_rhs(self, dfdt) -> torch.Tensor:
        """Assemble RHS of DIRK IMEX scheme for current stage.

        rhs = old[:,:,0]
            + sum_{j=0}^{istage-1} butcher_exp[istage, j] * expl[:,:,j]
            + sum_{j=0}^{istage-1} butcher_imp[istage, j] * impl[:,:,j]

        The Butcher table row index is istage (1-based), matching Fortran (istage+1).
        Column j is 0-based (Fortran j-1).
        """
        k = self.istage  # 1-based stage number
        rhs = dfdt.old[:, :, 0].clone()

        exp_row = self._exp_py[k]
        for j in range(k):
            w_exp = exp_row[j]
            if w_exp != 0.0:
                rhs = rhs + w_exp * dfdt.expl[:, :, j]

        imp_row = self._imp_py[k]
        for j in range(k):
            w_imp = imp_row[j]
            if w_imp != 0.0:
                rhs = rhs + w_imp * dfdt.impl[:, :, j]

        return rhs

    def set_imex_rhs_scalar(self, dfdt_scalar) -> float:
        """Assemble scalar RHS of DIRK IMEX scheme."""
        k = self.istage
        rhs = dfdt_scalar.old[0].item()

        exp_row = self._exp_py[k]
        for j in range(k):
            w_exp = exp_row[j]
            if w_exp != 0.0:
                rhs += w_exp * dfdt_scalar.expl[j].item()

        imp_row = self._imp_py[k]
        for j in range(k):
            w_imp = imp_row[j]
            if w_imp != 0.0:
                rhs += w_imp * dfdt_scalar.impl[j].item()

        return rhs

    def set_imex_rhs_multi(self, dfdts) -> torch.Tensor:
        """Assemble stacked RHS for multiple dfdts at once.

        Reduces MPS dispatch overhead by operating on larger tensors.
        Returns (n * lm_max, n_r_max) with results stacked vertically.
        """
        k = self.istage
        rhs = torch.cat([dt.old[:, :, 0] for dt in dfdts], dim=0).clone()

        exp_row = self._exp_py[k]
        for j in range(k):
            w = exp_row[j]
            if w != 0.0:
                rhs = rhs + w * torch.cat([dt.expl[:, :, j] for dt in dfdts], dim=0)

        imp_row = self._imp_py[k]
        for j in range(k):
            w = imp_row[j]
            if w != 0.0:
                rhs = rhs + w * torch.cat([dt.impl[:, :, j] for dt in dfdts], dim=0)

        return rhs

    def rotate_imex(self, dfdt):
        """No-op for DIRK schemes (stage slots are overwritten each step)."""
        pass

    def rotate_imex_scalar(self, dfdt_scalar):
        """No-op for DIRK schemes."""
        pass


# Module-level singleton — selected by params.time_scheme
def _create_tscheme():
    from .params import time_scheme
    if time_scheme == "CNAB2":
        return CNAB2()
    elif time_scheme == "BPR353":
        return BPR353()
    else:
        raise ValueError(f"Unknown time scheme: {time_scheme!r}. "
                         f"Supported: CNAB2, BPR353")


tscheme = _create_tscheme()
