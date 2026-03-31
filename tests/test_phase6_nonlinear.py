"""Phase 6: Verify nonlinear products against Fortran reference.

Runs 2 time steps (matching Fortran n_time_steps=2), then reproduces the
radial_loop internals from step 2 to compare intermediate grid-space fields
and combined advection + Lorentz force products.

Reference data at nR=17 (Fortran 1-indexed = Python nR=16):
- Grid-space B fields: brc/btc/bpc/cbrc/cbtc/cbpc
- Advection: Advr (pure), LFr (Lorentz radial)

NOTE: Fortran get_nl stores Advr and LFr as SEPARATE arrays. Our Python
get_nl combines advection + Lorentz into Advr/Advt/Advp. Tests compare
the radial component against (Advr + LFr). Advt/Advp cannot be tested
here because no LFt/LFp reference dumps exist.
"""

import torch
from conftest import load_ref
from magic_torch.params import (
    n_r_max, lm_max, n_theta_max, n_phi_max, dtmax,
)
from magic_torch.precision import DTYPE, CDTYPE, DEVICE
from magic_torch.radial_functions import or2
from magic_torch.horizontal_data import dLh

# Fortran nR=17 (1-indexed) = Python nR=16 (0-indexed)
NR17 = 16

_state = {}


def _ensure_init():
    if _state:
        return

    from magic_torch.init_fields import initialize_fields
    from magic_torch.step_time import setup_initial_state, initialize_dt, one_step
    from magic_torch.time_scheme import tscheme
    from magic_torch import fields
    from magic_torch.sht import (scal_to_spat, torpol_to_spat, torpol_to_curl_spat)
    from magic_torch.get_nl import get_nl

    # Run 2 steps to match Fortran (n_time_steps=2)
    initialize_fields()
    setup_initial_state()
    initialize_dt(dtmax)
    one_step(n_time_step=1, dt=dtmax)

    # Simulate step 2 dt rotation and weights (no-op for constant dt)
    dt_old = tscheme.dt[0].item()
    tscheme.dt[0] = dtmax
    tscheme.dt[1] = dt_old
    tscheme.set_weights()

    # Reproduce step 2's radial_loop inverse SHT + get_nl
    _dLh_1d = dLh.to(CDTYPE)
    f = fields
    N = n_r_max

    vrc = torch.zeros(N, n_theta_max, n_phi_max, dtype=DTYPE, device=DEVICE)
    vtc = torch.zeros_like(vrc)
    vpc = torch.zeros_like(vrc)
    cvrc = torch.zeros_like(vrc)
    cvtc = torch.zeros_like(vrc)
    cvpc = torch.zeros_like(vrc)
    sc = torch.zeros_like(vrc)
    brc = torch.zeros_like(vrc)
    btc = torch.zeros_like(vrc)
    bpc = torch.zeros_like(vrc)
    cbrc = torch.zeros_like(vrc)
    cbtc = torch.zeros_like(vrc)
    cbpc = torch.zeros_like(vrc)

    for nR in range(N):
        nBc = 2 if (nR == 0 or nR == N - 1) else 0
        if nBc == 0:
            sc[nR] = scal_to_spat(f.s_LMloc[:, nR])
            vrc[nR], vtc[nR], vpc[nR] = torpol_to_spat(
                _dLh_1d * f.w_LMloc[:, nR], f.dw_LMloc[:, nR], f.z_LMloc[:, nR])
            cvrc[nR], cvtc[nR], cvpc[nR] = torpol_to_curl_spat(
                or2[nR].item(), f.w_LMloc[:, nR], f.ddw_LMloc[:, nR],
                f.z_LMloc[:, nR], f.dz_LMloc[:, nR])
            brc[nR], btc[nR], bpc[nR] = torpol_to_spat(
                _dLh_1d * f.b_LMloc[:, nR], f.db_LMloc[:, nR], f.aj_LMloc[:, nR])
            cbrc[nR], cbtc[nR], cbpc[nR] = torpol_to_curl_spat(
                or2[nR].item(), f.b_LMloc[:, nR], f.ddb_LMloc[:, nR],
                f.aj_LMloc[:, nR], f.dj_LMloc[:, nR])
        else:
            brc[nR], btc[nR], bpc[nR] = torpol_to_spat(
                _dLh_1d * f.b_LMloc[:, nR], f.db_LMloc[:, nR], f.aj_LMloc[:, nR])
            cbrc[nR], cbtc[nR], cbpc[nR] = torpol_to_curl_spat(
                or2[nR].item(), f.b_LMloc[:, nR], f.ddb_LMloc[:, nR],
                f.aj_LMloc[:, nR], f.dj_LMloc[:, nR])
            sc[nR] = scal_to_spat(f.s_LMloc[:, nR])

    xic = torch.zeros_like(sc)  # no composition for dynamo benchmark
    (Advr, Advt, Advp, VSr, VSt, VSp,
     VxBr, VxBt, VxBp, VXir, VXit, VXip) = get_nl(
        vrc, vtc, vpc, cvrc, cvtc, cvpc,
        sc, brc, btc, bpc, cbrc, cbtc, cbpc, xic)

    _state.update({
        'brc': brc, 'btc': btc, 'bpc': bpc,
        'cbrc': cbrc, 'cbtc': cbtc, 'cbpc': cbpc,
        'Advr': Advr,
    })


# === Grid-space magnetic field tests at nR=17 ===

def test_brc_nR17():
    _ensure_init()
    ref = load_ref("brc_nR17")
    torch.testing.assert_close(_state['brc'][NR17], ref, atol=1e-10, rtol=1e-10)


def test_btc_nR17():
    _ensure_init()
    ref = load_ref("btc_nR17")
    torch.testing.assert_close(_state['btc'][NR17], ref, atol=1e-10, rtol=1e-10)


def test_bpc_nR17():
    _ensure_init()
    ref = load_ref("bpc_nR17")
    torch.testing.assert_close(_state['bpc'][NR17], ref, atol=1e-10, rtol=1e-10)


def test_cbrc_nR17():
    _ensure_init()
    ref = load_ref("cbrc_nR17")
    torch.testing.assert_close(_state['cbrc'][NR17], ref, atol=1e-10, rtol=1e-10)


def test_cbtc_nR17():
    _ensure_init()
    ref = load_ref("cbtc_nR17")
    torch.testing.assert_close(_state['cbtc'][NR17], ref, atol=1e-10, rtol=1e-10)


def test_cbpc_nR17():
    _ensure_init()
    ref = load_ref("cbpc_nR17")
    torch.testing.assert_close(_state['cbpc'][NR17], ref, atol=1e-10, rtol=1e-10)


# === Nonlinear product tests at nR=17 ===
# NOTE: Fortran Advr and LFr are separate. Our Advr = advection + Lorentz.
# Compare Advr against Fortran (Advr + LFr).

def test_Advr_nR17():
    """Our Advr includes Lorentz force. Compare against Fortran Advr + LFr."""
    _ensure_init()
    ref_advr = load_ref("Advr_nR17")
    ref_lfr = load_ref("LFr_nR17")
    ref_combined = ref_advr + ref_lfr
    torch.testing.assert_close(_state['Advr'][NR17], ref_combined, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                func()
                print(f"  {name} passed")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
    print("Phase 6: nonlinear tests done")
