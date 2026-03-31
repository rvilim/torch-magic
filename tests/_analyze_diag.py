"""Analyze the diagnostic data from _bouss_matrix_diag.py"""
import numpy as np
import os

d = '/tmp/bouss_matrix_diag'

# Load IC RHS components
ic_rhs_b = np.load(os.path.join(d, 'ic_rhs_b_stage1.npy'))
ic_rhs_j = np.load(os.path.join(d, 'ic_rhs_j_stage1.npy'))

# Load IC time array components
dbdt_ic_old = np.load(os.path.join(d, 'dbdt_ic_old_init.npy'))
dbdt_ic_impl = np.load(os.path.join(d, 'dbdt_ic_impl_init.npy'))
dbdt_ic_expl = np.load(os.path.join(d, 'dbdt_ic_expl_stage1.npy'))

djdt_ic_old = np.load(os.path.join(d, 'djdt_ic_old_init.npy'))
djdt_ic_impl = np.load(os.path.join(d, 'djdt_ic_impl_init.npy'))
djdt_ic_expl = np.load(os.path.join(d, 'djdt_ic_expl_stage1.npy'))

print("=== IC RHS shapes ===")
print(f"ic_rhs_b: {ic_rhs_b.shape}, ic_rhs_j: {ic_rhs_j.shape}")
print(f"dbdt_ic_old: {dbdt_ic_old.shape}")
print(f"dbdt_ic_impl: {dbdt_ic_impl.shape}")
print(f"dbdt_ic_expl: {dbdt_ic_expl.shape}")

print("\n=== IC RHS at ICB (index 0) - should be zero or very small ===")
print(f"ic_rhs_b[:, 0] max: {np.abs(ic_rhs_b[:, 0]).max():.6e}")
print(f"ic_rhs_j[:, 0] max: {np.abs(ic_rhs_j[:, 0]).max():.6e}")

print("\n=== IC RHS at bulk (index 1) ===")
print(f"ic_rhs_b[:, 1] max: {np.abs(ic_rhs_b[:, 1]).max():.6e}")
print(f"ic_rhs_j[:, 1] max: {np.abs(ic_rhs_j[:, 1]).max():.6e}")

print("\n=== IC old term at ICB (index 0) ===")
print(f"dbdt_ic_old[:, 0, 0] max: {np.abs(dbdt_ic_old[:, 0, 0]).max():.6e}")
print(f"djdt_ic_old[:, 0, 0] max: {np.abs(djdt_ic_old[:, 0, 0]).max():.6e}")

print("\n=== IC impl term at ICB (index 0) ===")
print(f"dbdt_ic_impl[:, 0, 0] max: {np.abs(dbdt_ic_impl[:, 0, 0]).max():.6e}")
print(f"djdt_ic_impl[:, 0, 0] max: {np.abs(djdt_ic_impl[:, 0, 0]).max():.6e}")

print("\n=== IC expl stage1 at ICB (index 0) ===")
print(f"dbdt_ic_expl[:, 0, 0] max: {np.abs(dbdt_ic_expl[:, 0, 0]).max():.6e}")
print(f"djdt_ic_expl[:, 0, 0] max: {np.abs(djdt_ic_expl[:, 0, 0]).max():.6e}")

# Manually verify the IC RHS:
# For BPR353 at stage 1: rhs = old[:,:,0] + butcher_exp[1,0]*expl[:,:,0] + butcher_imp[1,0]*impl[:,:,0]
# butcher_exp[1,0] = 1.0 * dt = 2e-4
# butcher_imp[1,0] = 0.5 * dt = 1e-4
dt = 2.0e-4
manual_rhs_b = dbdt_ic_old[:, :, 0] + dt * dbdt_ic_expl[:, :, 0] + 0.5*dt * dbdt_ic_impl[:, :, 0]
manual_rhs_j = djdt_ic_old[:, :, 0] + dt * djdt_ic_expl[:, :, 0] + 0.5*dt * djdt_ic_impl[:, :, 0]

print("\n=== Manual vs computed IC RHS ===")
diff_b = np.abs(manual_rhs_b - ic_rhs_b).max()
diff_j = np.abs(manual_rhs_j - ic_rhs_j).max()
print(f"max diff b: {diff_b:.6e}")
print(f"max diff j: {diff_j:.6e}")

# Now check the stage 1 fields against reference
ref_dir = '/Users/rvilim/dynamo/magic/samples/boussBenchSat/fortran_ref'

# Load reference step1 data
for name in ('aj_ic', 'dj_ic', 'ddj_ic', 'b_ic', 'db_ic', 'ddb_ic'):
    ref = np.load(os.path.join(ref_dir, f'{name}_step1.npy'))
    py = np.load(os.path.join(d, f'{name}_stage1.npy'))
    # Note: stage1 != step1 for DIRK (step1 = after all 4 stages)
    # So stage1 won't match step1 reference. This is expected.
    print(f"\n{name} stage1 vs step1 ref: shape py={py.shape}, ref={ref.shape}")
    if py.shape == ref.shape:
        maxabs = max(np.abs(ref).max(), 1e-30)
        relerr = np.abs(py - ref).max() / maxabs
        print(f"  rel err vs step1: {relerr:.6e} (expected to be large, different stages)")

# Check IC RHS: is the ICB surface row nonzero?
print("\n=== IC old at all radii (lm=10 as example) ===")
lm = 10
for nr in range(min(5, ic_rhs_b.shape[1])):
    print(f"  ic_rhs_b[{lm},{nr}] = {ic_rhs_b[lm, nr]:.6e}")
    print(f"  ic_rhs_j[{lm},{nr}] = {ic_rhs_j[lm, nr]:.6e}")

# Check initial fields
print("\n=== IC initial fields (from checkpoint) ===")
for name in ('b_ic', 'aj_ic'):
    ref_init = np.load(os.path.join(ref_dir, f'{name}_init.npy'))
    print(f"{name}_init max: {np.abs(ref_init).max():.6e}")

# Check omega_ic
omega_stage1 = np.load(os.path.join(d, 'omega_ic_stage1.npy'))
print(f"\nomega_ic after stage 1: {omega_stage1}")

# Check OC RHS for z (which has 1.6e-2 error at step1)
oc_rhs_b = np.load(os.path.join(d, 'oc_rhs_b_stage1.npy'))
oc_rhs_j = np.load(os.path.join(d, 'oc_rhs_j_stage1.npy'))
print(f"\n=== OC RHS stage1 ===")
print(f"oc_rhs_b max: {np.abs(oc_rhs_b).max():.6e}")
print(f"oc_rhs_j max: {np.abs(oc_rhs_j).max():.6e}")

# Check OC explicit terms
for name in ('dzdt', 'dwdt', 'dsdt', 'dpdt'):
    expl = np.load(os.path.join(d, f'{name}_expl_stage1.npy'))
    old = np.load(os.path.join(d, f'{name}_old_init.npy'))
    impl = np.load(os.path.join(d, f'{name}_impl_init.npy'))
    print(f"\n{name}:")
    print(f"  old[:,:,0] max: {np.abs(old[:,:,0]).max():.6e}")
    print(f"  impl[:,:,0] max: {np.abs(impl[:,:,0]).max():.6e}")
    print(f"  expl[:,:,0] max: {np.abs(expl[:,:,0]).max():.6e}")
