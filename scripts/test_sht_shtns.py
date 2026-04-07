#!/usr/bin/env python3
"""Unit tests: SHTns GPU wrapper vs bmm baseline.

Validates each of the 4 SHT wrapper functions individually for correctness.
Tests roundtrip (synthesis → analysis → synthesis), symmetry properties,
and direct comparison against the bmm implementation.

Usage:
    MAGIC_MODAL_GPU=B200 uv run modal run scripts/test_sht_shtns.py --lmax 128
"""

import os
import modal

GPU_TYPE = os.environ.get("MAGIC_MODAL_GPU", "H100")
SHTNS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "master")

app = modal.App("sht-shtns-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libfftw3-dev", "build-essential")
    .pip_install("torch", "numpy")
    .add_local_dir(SHTNS_SRC, remote_path="/root/shtns_src", copy=True)
    .run_commands("cd /root/shtns_src && CC=gcc CUDA_PATH=/usr/local/cuda pip install .")
    .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
                   remote_path="/root/src")
)


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def run_tests(lmax: int):
    import sys
    sys.path.insert(0, "/root/src")

    os.environ["MAGIC_LMAX"] = str(lmax)
    os.environ["MAGIC_DEVICE"] = "cuda"
    os.environ["MAGIC_USE_SHTNS"] = "0"  # import bmm version first

    import torch
    from magic_torch.params import lm_max, n_theta_max, n_phi_max
    from magic_torch.precision import CDTYPE, DTYPE, DEVICE
    from magic_torch.sht import (scal_to_spat as bmm_scal_to_spat,
                                  scal_to_SH as bmm_scal_to_SH,
                                  torpol_to_spat as bmm_torpol_to_spat,
                                  spat_to_sphertor as bmm_spat_to_sphertor)
    import magic_torch.sht_shtns as _sht_shtns_mod
    print(f"sht_shtns version: {_sht_shtns_mod.__doc__[:60]}")
    from magic_torch.sht_shtns import (scal_to_spat as shtns_scal_to_spat,
                                        scal_to_SH as shtns_scal_to_SH,
                                        torpol_to_spat as shtns_torpol_to_spat,
                                        spat_to_sphertor as shtns_spat_to_sphertor)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"l_max={lmax}, lm_max={lm_max}, grid={n_theta_max}x{n_phi_max}")

    torch.manual_seed(42)
    n_pass = 0
    n_fail = 0

    def check(name, ref, out, atol=1e-10, rtol=1e-10):
        nonlocal n_pass, n_fail
        if isinstance(ref, tuple):
            for i, (r, o) in enumerate(zip(ref, out)):
                check(f"{name}[{i}]", r, o, atol, rtol)
            return
        abs_err = (ref - out).abs().max().item()
        rel_err = abs_err / ref.abs().max().item() if ref.abs().max().item() > 0 else 0
        ok = abs_err < atol or rel_err < rtol
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        else:
            n_pass += 1
        print(f"  {status}: {name}: abs={abs_err:.2e}, rel={rel_err:.2e}")

    # --- Random spectral data ---
    def rand_spec(nb):
        s = (torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)
             + 1j * torch.randn(lm_max, nb, dtype=DTYPE, device=DEVICE)).to(CDTYPE)
        s[0] = s[0].real.clone()  # m=0 must be real for real spatial fields
        return s

    # === Test 1: scal_to_spat ===
    print("\n--- scal_to_spat ---")
    for nb in [1, 32]:
        Slm = rand_spec(nb)
        if nb == 1:
            Slm = Slm.squeeze(1)
        ref = bmm_scal_to_spat(Slm)
        out = shtns_scal_to_spat(Slm)
        check(f"scal_to_spat(nb={nb})", ref, out)

    # === Test 2: scal_to_SH ===
    print("\n--- scal_to_SH ---")
    for nb in [1, 32]:
        Slm = rand_spec(nb)
        if nb == 1:
            Slm = Slm.squeeze(1)
        grid = bmm_scal_to_spat(Slm)
        ref = bmm_scal_to_SH(grid)
        out = shtns_scal_to_SH(grid)
        check(f"scal_to_SH(nb={nb})", ref, out)

    # === Test 3: scal roundtrip ===
    print("\n--- scal roundtrip ---")
    Slm = rand_spec(32)
    grid = shtns_scal_to_spat(Slm)
    Slm_back = shtns_scal_to_SH(grid)
    err = (Slm - Slm_back).abs().max().item()
    ok = err < 1e-10
    status = "PASS" if ok else "FAIL"
    if ok: n_pass += 1
    else: n_fail += 1
    print(f"  {status}: roundtrip error: {err:.2e}")
    if not ok:
        # Diagnostic: check if it's a constant factor
        mask = Slm.abs() > Slm.abs().max() * 0.01
        if mask.sum() > 0:
            ratio = (Slm_back[mask] / Slm[mask]).abs()
            print(f"    ratio (back/orig): mean={ratio.mean():.6f}, std={ratio.std():.6f}")
        # Also try bmm roundtrip for comparison
        grid_bmm = bmm_scal_to_spat(Slm)
        Slm_back_bmm = bmm_scal_to_SH(grid_bmm)
        err_bmm = (Slm - Slm_back_bmm).abs().max().item()
        print(f"    bmm roundtrip error: {err_bmm:.2e}")
        # And cross: shtns synthesis → bmm analysis
        Slm_cross = bmm_scal_to_SH(grid)
        err_cross = (Slm - Slm_cross).abs().max().item()
        print(f"    cross (shtns synth → bmm anal): {err_cross:.2e}")

    # === Test 4: torpol_to_spat ===
    print("\n--- torpol_to_spat ---")
    for nb in [1, 32]:
        Q = rand_spec(nb)
        S = rand_spec(nb)
        T = rand_spec(nb)
        if nb == 1:
            Q, S, T = Q.squeeze(1), S.squeeze(1), T.squeeze(1)
        ref = bmm_torpol_to_spat(Q, S, T)
        out = shtns_torpol_to_spat(Q, S, T)
        check(f"torpol_to_spat(nb={nb})", ref, out)
        if nb == 1:
            # Diagnostic: check if angular components are related by a simple factor
            for comp, name in [(1, "bt"), (2, "bp")]:
                r = ref[comp].flatten()
                o = out[comp].flatten()
                mask = r.abs() > r.abs().max() * 0.01
                if mask.sum() > 0:
                    ratio = (o[mask] / r[mask])
                    print(f"    {name} ratio (shtns/bmm) mean={ratio.mean():.4f} std={ratio.std():.4f}")

    # === Test 5: spat_to_sphertor ===
    print("\n--- spat_to_sphertor ---")
    for nb in [1, 32]:
        Q = rand_spec(nb)
        S = rand_spec(nb)
        T = rand_spec(nb)
        if nb == 1:
            Q, S, T = Q.squeeze(1), S.squeeze(1), T.squeeze(1)
        brc, btc, bpc = bmm_torpol_to_spat(Q, S, T)
        ref = bmm_spat_to_sphertor(btc, bpc)
        out = shtns_spat_to_sphertor(btc, bpc)
        check(f"spat_to_sphertor(nb={nb})", ref, out)

    # === Test 6: Direct SHTns sphtor synthesis vs bmm ===
    print("\n--- sphtor synthesis diagnostic ---")
    from magic_torch.sht import sphtor_to_spat as bmm_sphtor_to_spat
    S_test = rand_spec(1).squeeze(1)
    T_test = rand_spec(1).squeeze(1)
    # bmm sphtor_to_spat
    vt_bmm, vp_bmm = bmm_sphtor_to_spat(S_test, T_test)
    # SHTns: use cu_SHsphtor_to_spat directly
    sh_diag = shtns_scal_to_spat  # just to get a config
    sh1 = shtns_torpol_to_spat  # trigger config creation
    from magic_torch.sht_shtns import _get_config, _spat_from_shtns, _spec_to_shtns, _grid_idx
    sh_cfg = _get_config(1)
    s_gpu = _spec_to_shtns(S_test.unsqueeze(1))
    t_gpu = _spec_to_shtns(T_test.unsqueeze(1))
    vt_out = torch.empty(1, sh_cfg.nphi, sh_cfg.nlat, dtype=DTYPE, device=DEVICE)
    vp_out = torch.empty_like(vt_out)
    torch.cuda.synchronize()
    sh_cfg.cu_SHsphtor_to_spat(s_gpu.data_ptr(), t_gpu.data_ptr(),
                                 vt_out.data_ptr(), vp_out.data_ptr())
    torch.cuda.synchronize()
    # Raw SHTns output (sorted, no Robert form)
    vt_sorted = vt_out.transpose(-1, -2)  # (1, n_theta_sorted, n_phi)
    vt_raw = vt_sorted[0, _grid_idx, :]
    err_raw = (vt_raw - vt_bmm).abs().max().item()
    print(f"  vt raw: err={err_raw:.2e}")
    print(f"  vt bmm range: [{vt_bmm.min():.2e}, {vt_bmm.max():.2e}]")
    print(f"  vt shtns raw range: [{vt_raw.min():.2e}, {vt_raw.max():.2e}]")

    # === Test 7: Y_1^0 theta ordering verification ===
    print("\n--- Y_1^0 theta ordering ---")
    Slm_y10 = torch.zeros(lm_max, dtype=CDTYPE, device=DEVICE)
    Slm_y10[1] = 1.0  # l=1, m=0
    grid_bmm = bmm_scal_to_spat(Slm_y10)
    grid_shtns = shtns_scal_to_spat(Slm_y10)
    err = (grid_bmm - grid_shtns).abs().max().item()
    ok = err < 1e-12
    status = "PASS" if ok else "FAIL"
    if ok: n_pass += 1
    else: n_fail += 1
    print(f"  {status}: Y_1^0 max diff: {err:.2e}")
    # Verify the pattern: Y_1^0 should be positive in N hemisphere, negative in S
    mid = n_theta_max // 2
    north_mean = grid_shtns[:mid].mean().item()
    south_mean = grid_shtns[mid:].mean().item()
    print(f"  Y_1^0 N-hemisphere mean: {north_mean:.4f}, S-hemisphere mean: {south_mean:.4f}")

    # === Summary ===
    print(f"\n{'='*40}")
    print(f"Results: {n_pass} passed, {n_fail} failed")
    if n_fail > 0:
        raise AssertionError(f"{n_fail} tests failed!")
    print("All tests passed!")


@app.local_entrypoint()
def main(lmax: int = 128):
    print(f"Running SHTns unit tests: l_max={lmax}, GPU={GPU_TYPE}")
    run_tests.remote(lmax)
