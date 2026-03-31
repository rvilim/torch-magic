# Plan: G_1.TAG Binary Graph Output File

## Context

G_1.TAG is a binary stream file containing 3D grid-space fields (velocity, entropy, pressure, magnetic field) at a snapshot time. For dynamo_benchmark: `n_graphs=1`, so one file written at the last time step (step 3, time=3e-4).

Reference: `samples/dynamo_benchmark/G_1.test` (1,451,968 bytes, big-endian float32).

---

## File Structure

### Header (448 bytes for benchmark)
Written as raw stream (no Fortran record markers):

| Field | Type | Bytes |
|-------|------|-------|
| version (=14) | int32 BE | 4 |
| runid (64 chars) | char[64] | 64 |
| time | float32 BE | 4 |
| ra, pr, raxi, sc, ek, stef, prmag, radratio, sigma_ratio | 9×float32 BE | 36 |
| n_r_max, n_theta_max, n_phi_tot, minc, n_r_ic_max | 5×int32 BE | 20 |
| l_heat, l_chemical_conv, l_phase_field, l_mag, l_PressGraph, l_cond_ic | 6×int32 BE (logical) | 24 |
| theta_ord(1:n_theta_max) | n_theta_max × float32 BE | 96 |
| r(1:n_r_max) | n_r_max × float32 BE | 132 |
| r_ic(1:n_r_ic_max) | n_r_ic_max × float32 BE | 68 (if l_mag and n_r_ic_max>1) |

### OC Data (per radial level, nR=0..n_r_max-1)
Each field is (n_theta_max, n_phi_max) float32 BE = 4608 bytes.
Fields per level (for benchmark: l_heat=T, l_chem=F, l_phase=F, l_PressGraph=T, l_mag=T):
1. vr = or2[nR] * vScale * orho1[nR] * vr_grid
2. vt = or1[nR] * vScale * orho1[nR] * O_sin_theta * vt_grid
3. vp = or1[nR] * vScale * orho1[nR] * O_sin_theta * vp_grid
4. sr = sr_grid (entropy, no scaling)
5. prer = prer_grid (pressure, no scaling) — since l_PressGraph=True
6. br = or2[nR] * br_grid
7. bt = or1[nR] * O_sin_theta * bt_grid
8. bp = or1[nR] * O_sin_theta * bp_grid

**Theta reordering**: SHT output is interleaved N/S: [θ_N[0], θ_S[0], θ_N[1], θ_S[1], ...]. Graph file uses geographic order (north-to-south, matching theta_ord). The mapping `n_theta_cal2ord`:
- interleaved index 2k → geographic index k (north half)
- interleaved index 2k+1 → geographic index n_theta_max-1-k (south half)

### IC Data (per IC radial level, nR=0..n_r_ic_max-1)
For insulating IC (l_cond_ic=False): compute potential field from bICB = b_LMloc[:, -1].
Uses `torpol_to_spat_IC(r_ic[nR], r_ICB, bICB, ...)` where Wlm=bICB for all levels.

Per level, 3 fields (n_theta_max, n_phi_max) float32 BE:
1. Br_ic = BrB * O_r_ic2[nR]
2. Bt_ic = BtB * O_r_ic[nR] * O_sin_theta
3. Bp_ic = BpB * O_r_ic[nR] * O_sin_theta

Where O_r_ic = 1/r_ic, O_r_ic2 = 1/r_ic^2.

---

## Step 1: Add `l_PressGraph` to params.py

```python
l_PressGraph = True  # Namelists.f90:1641, always set true
```

---

## Step 2: Build theta reorder permutation in horizontal_data.py

```python
# Mapping from interleaved (calculation) order to geographic (sorted) order
# Fortran: n_theta_cal2ord(2k-1) = k, n_theta_cal2ord(2k) = n_theta_max-k+1
# Python 0-indexed: interleaved[2k] → geo[k], interleaved[2k+1] → geo[n_theta_max-1-k]
n_theta_cal2ord = torch.zeros(n_theta_max, dtype=torch.long, device=DEVICE)
for k in range(_NHS):
    n_theta_cal2ord[2 * k] = k           # north hemisphere
    n_theta_cal2ord[2 * k + 1] = n_theta_max - 1 - k  # south hemisphere
```

This is the INVERSE of `_grid_idx`. Verification: `_grid_idx` maps geo→interleaved; `n_theta_cal2ord` maps interleaved→geo. So `n_theta_cal2ord[_grid_idx] == arange(n_theta_max)`.

---

## Step 3: Add `torpol_to_spat_IC` to sht.py

Port the IC SHT from `sht_native.f90:188-229`. For insulating IC:
```python
def torpol_to_spat_IC(r_val, r_ICB, Wlm, dWlm, Zlm):
    """IC potential field SHT. Wlm=bICB for insulating IC."""
    rRatio = r_val / r_ICB
    # rDep[l] = (r/r_ICB)^(l+1), rDep2[l] = (r/r_ICB)^l / r_ICB
    rDep = rRatio ** (torch.arange(l_max+1) + 1)   # (l_max+1,)
    rDep2 = rRatio ** torch.arange(l_max+1) / r_ICB

    # Expand to lm_max via st_lm2l
    Qlm = rDep[st_lm2l] * dLh * Wlm
    Slm = rDep2[st_lm2l] * ((st_lm2l + 1) * Wlm + r_val * dWlm)
    Tlm = rDep[st_lm2l] * Zlm
    return native_qst_to_spat(Qlm, Slm, Tlm)  # reuse existing SHT
```

For insulating IC, dWlm and Zlm are both from `db_ic[:,0]` and `aj_ic[:,0]` (only ICB values used). Actually, looking at Fortran line 667: for insulating IC, `dWlm = db_ic(:,1)` and `Zlm = aj_ic(:,1)`. Since we DON'T store IC fields when l_cond_ic=False...

Wait. Looking at the fields.py: `b_ic, db_ic, aj_ic` are always allocated. For insulating IC, `db_ic` and `aj_ic` are zero-initialized and never updated. The key is `bICB = b[:, -1]` (OC b at ICB). So for insulating IC, `torpol_to_spat_IC` is called with `Wlm=bICB, dWlm=db_ic[:,0]=0, Zlm=aj_ic[:,0]=0`, meaning `Slm = rDep2 * (l+1) * bICB` and `Tlm = 0`.

Actually, re-reading: for insulating IC, the call is:
```fortran
call torpol_to_spat_IC(r_ic(nR), r_ICB, bICB(:), db_ic(:,1), aj_ic(:,1), BrB, BtB, BpB)
```
So dWlm = db_ic(:,1) and Zlm = aj_ic(:,1). These are the ICB values of the IC field derivatives. For insulating IC these are zero. But wait — the Fortran DOES initialize db_ic and aj_ic for insulating IC via potential field matching? Let me check...

Actually this is getting complex. For the dynamo_benchmark, l_cond_ic=False. The b_ic, db_ic, aj_ic arrays exist but for insulating IC the derivative db_ic and toroidal aj_ic at ICB may be non-trivially set from the potential field matching. Let me check what init does...

Actually, looking more carefully: for insulating IC, the Fortran just passes `db_ic(:,1)` and `aj_ic(:,1)`. These are zero for insulating IC since the arrays are allocated but never filled with meaningful data. The only non-zero input is `bICB = b[:, n_r_max]` from the OC field at ICB. Inside torpol_to_spat_IC:
- `Qlm = rDep[l] * dLh * bICB`
- `Slm = rDep2[l] * ((l+1)*bICB + r*0)` = `rDep2[l] * (l+1) * bICB`
- `Tlm = rDep[l] * 0` = 0

So it's purely a poloidal potential field from the OC b at ICB. No toroidal component.

---

## Step 4: New file `graph_output.py`

```python
def write_graph_file(path, time, fields):
    """Write G_1.TAG binary graph file matching Fortran format."""
```

### 4a. Header writer
Write all header fields as big-endian binary (struct.pack with '>').

### 4b. OC field writer
For each nR in range(n_r_max):
1. Compute grid-space velocity: `torpol_to_spat(w[:,nR], dw[:,nR], z[:,nR])` → (vr, vt, vp)
2. Compute grid-space entropy: `scal_to_spat(s[:,nR])` → sr
3. Compute grid-space pressure: `scal_to_spat(p[:,nR])` → prer
4. Compute grid-space B-field: need br, bt, bp from `torpol_to_spat(b[:,nR], db[:,nR], aj[:,nR])`
   - Wait: torpol_to_spat gives (Qr, St, Sp) not (Br, Bt, Bp). For B-field, `Br = dLh*or2*b`, `Bt/Bp` from the poloidal/toroidal decomposition. Actually our existing `torpol_to_spat` already does this — it takes (Qlm=dLh*or2*Blm, Slm=dBlm, Tlm=AJlm) and returns (br, bt, bp) in grid space.

Actually, looking at how the Fortran radial loop (rIter.f90) computes these: the SHT input for B-field is via `torpol_to_spat` with the same spectral coefficients that the velocity uses. The difference is just the physical meaning and scaling.

For the magnetic field, we need the same SHT as velocity (QST → spatial):
- Input: Qlm = dLh * Blm (not or2*dLh*Blm — the or2 is applied AFTER in graphOut)

Hmm, let me look at our existing `torpol_to_spat` more carefully to understand what it returns.

### Key insight: reuse existing batched SHT
Our `torpol_to_spat` already handles batched radial levels. We can call it once for ALL radial levels for velocity and once for magnetic field, then apply the scaling factors.

The SHT outputs (vr, vt, vp) are in interleaved theta order. We reorder to geographic using `n_theta_cal2ord`.

### 4c. IC field writer
For each IC radial level:
1. Call `torpol_to_spat_IC(r_ic[nR], r_ICB, bICB, zeros, zeros)`
2. Apply scaling: Br *= O_r_ic2, Bt *= O_r_ic * O_sin_theta, Bp *= O_r_ic * O_sin_theta
3. Reorder theta, cast to float32 BE, write

### 4d. Binary writing
All float data cast to float32, all written big-endian using `numpy` with `>f4` dtype or `struct.pack`.

---

## Step 5: Integration into main.py

Add `n_graphs` parameter (default 1). Compute graph output step number:
```python
n_graph_step = n_steps  # for n_graphs=1: output at the last step
```

In the time loop, when `step == n_graph_step`:
```python
write_graph_file(os.path.join(output_dir, f"G_1.{tag}"), sim_time, fields)
```

---

## Step 6: Tests

### test_G1_header
Parse the header of our G_1 output and compare against reference.

### test_G1_file_size
Compare total file size: should be exactly 1,451,968 bytes.

### test_G1_OC_fields
For each of the 8 OC field types, at selected radial levels (e.g., nR=0, 16, 32), compare against reference with rtol~1e-6 (float32 precision).

### test_G1_IC_fields
For selected IC radial levels, compare Br, Bt, Bp against reference.

---

## Files Changed

| File | Change |
|------|--------|
| `params.py` | Add `l_PressGraph = True` |
| `horizontal_data.py` | Add `n_theta_cal2ord` permutation array |
| `sht.py` | Add `torpol_to_spat_IC()` |
| `graph_output.py` (NEW) | `write_graph_file()` with header + OC + IC writers |
| `main.py` | Call `write_graph_file()` at graph step |
| `tests/test_graph_output.py` (NEW) | Header, file size, OC field, IC field tests |

---

## Critical Details

1. **Byte order**: All output is big-endian (Fortran default). Use `numpy.array(...).astype('>f4')` or `struct.pack('>...')`.
2. **No record markers**: Fortran `access='stream'` means no 4-byte length prefix/suffix. Raw bytes only.
3. **Theta reorder**: MUST reorder from interleaved to geographic before writing.
4. **l_PressGraph**: Always True (Namelists.f90:1641). Pressure field IS included.
5. **l_cond_ic=False**: IC potential field from OC b at ICB. db_ic and aj_ic are zero.
6. **O_sin_theta**: In interleaved (calculation) order, NOT geographic order. Apply BEFORE theta reorder.
7. **vScale = 1.0** for Boussinesq benchmark (lScale/tScale = 1).
8. **stef = 0.0** (Stefan number, not used in benchmark).
9. **Float32 precision**: Output is float32 even though computation is float64. Cast at write time.
10. **Graph timing**: `n_graphs=1` means output at the last log step. For dynamo_benchmark with 3 integrating steps, this is step 3 (time=3e-4).

## Open Questions

- The exact SHT call signature for magnetic field: does our `torpol_to_spat` take the same arguments as for velocity, or does the B-field need `torpol_to_curl_spat`? Need to verify against the Fortran radial loop to see which SHT function produces the br/bt/bp that graphOut receives.
