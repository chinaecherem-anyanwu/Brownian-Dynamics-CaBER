#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import this


# In[ ]:





# In[1]:


import os
import numpy as np

folder = "BD FENE data (different shear & elongational shear rates)"

case_tags = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"] #["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "E1", "E2", "E3", "E4", "E5", "E6", "E7"] # "E8"] #["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",  

S_list = []
dSdt_list = []
kappa_list = []
label_list = []

for tag in case_tags:
    S_file     = os.path.join(folder, f"fenep_raw_data_S_{tag}.npy")
    dSdt_file  = os.path.join(folder, f"fenep_raw_data_dS_dt_{tag}.npy")
    kappa_file = os.path.join(folder, f"fenep_raw_data_kappa_{tag}.npy")

    print(f"Loading case {tag}...")
    print("  S file:", S_file)

    S = np.load(S_file)
    dSdt = np.load(dSdt_file)
    kappa = np.load(kappa_file)

    N = S.shape[0]

    print(f"  S shape:     {S.shape}")
    print(f"  dSdt shape:  {dSdt.shape}")
    print(f"  kappa shape: {kappa.shape}")

    if S.shape != dSdt.shape:
        raise ValueError(f"Shape mismatch in case {tag}: S and dSdt do not match.")

    # Case 1: kappa is constant within this simulation
    if kappa.shape == (3, 3):
        kappa_field = np.broadcast_to(kappa, (N, 3, 3)).copy()

    # Case 2: kappa already changes with time
    elif kappa.shape == (N, 3, 3):
        kappa_field = kappa

    else:
        raise ValueError(
            f"kappa for case {tag} must have shape (3,3) or (N,3,3), "
            f"but got {kappa.shape}"
        )

    S_list.append(S)
    dSdt_list.append(dSdt)
    kappa_list.append(kappa_field)

    labels = np.array([tag] * N)
    label_list.append(labels)


S_all = np.concatenate(S_list, axis=0)
dSdt_all = np.concatenate(dSdt_list, axis=0)
kappa_all = np.concatenate(kappa_list, axis=0)
labels_all = np.concatenate(label_list, axis=0)

print("\nMerged shapes:")
print(f"S_all shape:      {S_all.shape}")
print(f"dSdt_all shape:   {dSdt_all.shape}")
print(f"kappa_all shape:  {kappa_all.shape}")
print(f"labels shape:     {labels_all.shape}")

np.save("fenep_merged_S.npy", S_all)
np.save("fenep_merged_dS_dt.npy", dSdt_all)
np.save("fenep_merged_kappa.npy", kappa_all)
np.save("fenep_merged_labels.npy", labels_all)

print("\nSaved merged files:")
print("  fenep_merged_S.npy")
print("  fenep_merged_dS_dt.npy")
print("  fenep_merged_kappa.npy")
print("  fenep_merged_labels.npy")


# In[ ]:





# In[56]:


import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ------------------------------------------------------------
# Helper function for symmetric tensors: S and dSdt
# Prints [xx, yy, zz, xy, xz, yz]
# ------------------------------------------------------------
def print_first_20_symmetric_components(name, A):
    comps = np.column_stack([
        A[:20, 0, 0],  # xx
        A[:20, 1, 1],  # yy
        A[:20, 2, 2],  # zz
        A[:20, 0, 1],  # xy
        A[:20, 0, 2],  # xz
        A[:20, 1, 2],  # yz
    ])

    print(f"\n{name}")
    print("First 20 entries as [xx, yy, zz, xy, xz, yz]:")
    print(comps)


# ------------------------------------------------------------
# Helper function for nonsymmetric tensor: kappa
# Prints all 9 components [xx, xy, xz, yx, yy, yz, zx, zy, zz]
# ------------------------------------------------------------
def print_first_20_all_kappa_components(name, A):
    comps = np.column_stack([
        A[:20, 0, 0],  # xx
        A[:20, 0, 1],  # xy
        A[:20, 0, 2],  # xz

        A[:20, 1, 0],  # yx
        A[:20, 1, 1],  # yy
        A[:20, 1, 2],  # yz

        A[:20, 2, 0],  # zx
        A[:20, 2, 1],  # zy
        A[:20, 2, 2],  # zz
    ])

    print(f"\n{name}")
    print("First 20 entries as [xx, xy, xz, yx, yy, yz, zx, zy, zz]:")
    print(comps)


# ------------------------------------------------------------
# Print S, dSdt, kappa, and labels for each case
# ------------------------------------------------------------
for tag, S, dSdt, kappa in zip(case_tags, S_list, dSdt_list, kappa_list):

    print("\n" + "="*80)
    print(f"CASE: {tag}")
    print("="*80)

    print_first_20_symmetric_components(f"S for case {tag}", S)
    print_first_20_symmetric_components(f"dSdt for case {tag}", dSdt)
    print_first_20_all_kappa_components(f"kappa for case {tag}", kappa)

    print(f"\nFirst 20 labels for case {tag}:")
    print(np.array([tag] * min(20, S.shape[0])))

    print("\nFirst full kappa matrix:")
    print(kappa[0])


# ------------------------------------------------------------
# Print S_all, dSdt_all, kappa_all, and labels_all
# ------------------------------------------------------------
print("\n" + "="*80)
print("MERGED DATA")
print("="*80)

print_first_20_symmetric_components("Merged S_all", S_all)
print_first_20_symmetric_components("Merged dSdt_all", dSdt_all)
print_first_20_all_kappa_components("Merged kappa_all", kappa_all)

print("\nFirst 20 merged labels:")
print(labels_all[:20])

print("\nFirst full merged kappa matrix:")
print(kappa_all[0])


# In[ ]:





# In[3]:


# ============================================================
# STRIDGE / SINDY-STYLE FENE CONFORMATION EQUATION RECOVERY
# ------------------------------------------------------------


# Import numpy for tensor algebra
import numpy as np

# Import pandas to save discovered coefficients
import pandas as pd


# -----------------------------
# 1. INPUT FILES FROM GROUND-TRUTH GENERATOR
# -----------------------------

# These names match the output names
S_npy     = "fenep_merged_S.npy"
dS_npy    = "fenep_merged_dS_dt.npy"
kappa_npy = "fenep_merged_kappa.npy"
labels_npy = "fenep_merged_labels.npy"

# Output filename for recovered coefficients
coefficients_csv = "stridge_fene_merged_unscaled_coefficients.csv"

# FENE finite-extensibility denominator constant.
# In the FENE, DENOM0 = 2R0^2.
DENOM0 = 20000.0


# -----------------------------
# 2. LOAD FENE-P TENSOR DATA
# -----------------------------

# FENE-P conformation tensor field, shape (N, 3, 3)
S = np.load(S_npy)

# Time derivative of the conformation tensor, shape (N, 3, 3)
dS_dt = np.load(dS_npy)

# Velocity-gradient tensor, shape (3, 3)
kappa = np.load(kappa_npy)

# Labels for case by case identification (N,1)
labels = np.load(labels_npy)

# Number of time points
N = S.shape[0]

# Print loaded array shapes for sanity check
print(f"Loaded merged data with {N} samples.")
print(f"S shape:       {S.shape}")
print(f"dS_dt shape:   {dS_dt.shape}")
print(f"kappa shape:   {kappa.shape}")
print(f"labels shape:  {labels.shape}")



# Basic checks
if S.shape != dS_dt.shape:
    raise ValueError("S and dS_dt must have the same shape.")

if kappa.shape != S.shape:
    raise ValueError(
        f"For merged data, kappa must have shape {S.shape}, but got {kappa.shape}"
    )

# -----------------------------
# 3. BUILD TENSORIAL CANDIDATE LIBRARY
# -----------------------------
# The library contains tensor terms that may appear in the FENE-P equation.
# Each term must have shape (N, 3, 3), because STRidge is fitting tensor
# components of dS/dt - UC.

# Unit tensor d = δ_ij
# This captures the constant isotropic source term in the FENE-P relaxation.
d = np.eye(3)
d_field = np.broadcast_to(d, (N, 3, 3))
d1 = 8.0 * d_field

# For merged data, kappa is already a field of shape (N,3,3)
kappa_field = kappa

# Direct kappa-transpose tensor field, shape (N, 3, 3)
# Included as another possible flow-gradient candidate term.
kappaT_field = np.transpose(kappa_field, (0, 2, 1))

# Tensor products involving S and kappa
S_kappa  = S @ kappa_field
S_kappaT = S @ kappaT_field
kappa_S  = kappa_field @ S
kappaT_S = kappaT_field @ S


S_dot_S = S @ S

# Lumped convected terms
# UCij = S·kappa + kappa^T·S
UCij = S_kappa + kappaT_S

# Trace of S, shape (N,)
# This appears in the FENE-P finite-extensibility denominator.
tr_S = np.trace(S, axis1=1, axis2=2)

# Likely FENE nonlinear spring term.
# This term represents:
#   S/(1 - tr(S)/(2R0^2))
# Since DENOM0 = 2R0^2, the denominator is:
#   1 - tr(S)/DENOM0
Sp = tr_S[:, None, None] / DENOM0
FENE_denominator = 1.0 - Sp
FENE_S1 = S / FENE_denominator
FENE_S2 = (Sp * S) / FENE_denominator
FENE_S3 = (Sp * Sp * S) / FENE_denominator
FENE_S4 = (Sp * Sp * Sp * S) / FENE_denominator
My_FENE = (60.834 * Sp * S) / (1 + (30.896 * Sp))
FENE_S_dot_S = (Sp * S_dot_S) / FENE_denominator

# Candidate tensor library for the first-order expanded FENE-P model.
# The expected active terms are:
#   d, S, S_kappa, kappaT_S, and S_tr_S.
# The other terms are included to test whether STRidge eliminates them.
library = {
    "S":      S,
    "FENE_S1": FENE_S1,
    "FENE_S2": FENE_S2,
    "FENE_S3": FENE_S3,
    #"FENE_S4": FENE_S4,
    #"My_FENE": My_FENE,
    "S_dot_S":  S_dot_S,
    "FENE_S_dot_S": FENE_S_dot_S,
}

# Store the candidate names and count them
term_names = list(library.keys())
n_terms = len(term_names)

# Print the candidate terms for inspection
print("\nCandidate library terms:")
for name in term_names:
    print(f"  {name}")

print(f"\nNumber of candidate terms: {n_terms}")


# -----------------------------
# 4. STACK TENSOR COMPONENTS INTO A REGRESSION MATRIX
# -----------------------------
# Each row corresponds to one (time, tensor component) pair.
# All 9 components of the 3×3 tensor are used.

component_indices = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
]

n_components = len(component_indices)

# Allocate feature matrix X and target vector y in raw physical units
X = np.zeros((N * n_components, n_terms))
y = np.zeros(N * n_components)

# Fill X and y row-by-row
for t_idx in range(N):
    for c_idx, (i, j) in enumerate(component_indices):
        row_idx = t_idx * n_components + c_idx

        # Fill each library feature column
        for k_idx, name in enumerate(term_names):
            X[row_idx, k_idx] = library[name][t_idx, i, j]

        # Target is the matching component of dS/dt - UC
        y[row_idx] = dS_dt[t_idx, i, j] - UCij[t_idx, i, j] - d1[t_idx, i, j]

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape:  {y.shape}")


# -----------------------------
# 5. DEFINE STRIDGE
# -----------------------------
# STRidge = Sequential Thresholded Ridge Regression.
#
# Same algorithm as the standardized version, but here it operates
# directly on raw physical-unit features. The threshold is now
# applied to physical-unit coefficients.

def stridge(X, y, lam=1e-8, threshold=1e-4, max_iter=20):

    #Sequential thresholded ridge regression.

    #Args:
        #X: feature matrix, shape (n_samples, n_features) — raw physical units
        #y: target vector, shape (n_samples,)             — raw physical units
        #lam: ridge regularization strength
        #threshold: coefficients smaller than this are removed
        #max_iter: maximum number of threshold/refit iterations

    #Returns:
        #xi: sparse coefficient vector in physical units


    n_features = X.shape[1]

    # Initial ridge regression using all terms
    XtX = X.T @ X + lam * np.eye(n_features)
    Xty = X.T @ y
    xi = np.linalg.solve(XtX, Xty)

    # Sequential thresholding loop
    for iteration in range(max_iter):

        # Identify small coefficients
        small = np.abs(xi) < threshold

        # Stop if all coefficients are small
        if small.all():
            break

        # Zero small coefficients
        xi[small] = 0.0

        # Active terms are the ones not zeroed out
        active = ~small

        # Stop if nothing remains active
        if active.sum() == 0:
            break

        # Refit only active terms
        X_active   = X[:, active]
        XtX_active = X_active.T @ X_active + lam * np.eye(active.sum())
        Xty_active = X_active.T @ y
        xi_active  = np.linalg.solve(XtX_active, Xty_active)

        # Store refit active coefficients
        xi[active] = xi_active

    return xi


# -----------------------------
# 6. FIT SPARSE MODEL ON RAW (UNSCALED) DATA
# -----------------------------
# No standardization, no back-transformation. The coefficients
# returned by STRidge are already in physical units.

xi_physical = stridge(
    X,
    y,
    lam=1e-8,
    threshold=4.5e-2,
    max_iter=100,
)


# -----------------------------
# 7. REPORT DISCOVERED EQUATION
# -----------------------------

print("\n" + "=" * 70)
print("STRIDGE DISCOVERED CONSTITUTIVE EQUATION (UNSCALED FIT)")
print("=" * 70)

print("\nForm:")
print("  dS/dt = UCij + 8*d + sum_k xi_k * T_k(S)")

print(f"\n{'Term':<20s} {'Coefficient (physical)':>25s}")
print("-" * 50)

active_terms = []

for k, name in enumerate(term_names):
    coeff = xi_physical[k]

    if np.abs(coeff) > 1e-8:
        print(f"{name:<20s} {coeff:>25.8f}")
        active_terms.append((name, coeff))
    else:
        print(f"{name:<20s} {'0 (eliminated)':>25s}")

print(f"\nNumber of active terms: {len(active_terms)} / {n_terms}")


# -----------------------------
# 8. FIT QUALITY
# -----------------------------

# Predict in physical scale (no intercept term)
y_pred = X @ xi_physical

residuals = y - y_pred

mse    = np.mean(residuals ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
ss_res = np.sum(residuals ** 2)
r2     = 1.0 - ss_res / ss_tot

print("\n" + "=" * 70)
print("FIT QUALITY")
print("=" * 70)
print(f"MSE: {mse:.8e}")
print(f"R²:  {r2:.8f}")


# -----------------------------
# 9. COMPARE AGAINST EXPECTED EXACT FENE-P FORM
# -----------------------------
# Starting from the FENE-P conformation equation:
#
#   dS/dt = S·kappa + kappa^T·S
#           - (1/lambda) * S/(1 - tr(S)/(2R0^2))
#           + (1/lambda)*beta^-2*d
#
# Define:
#
#   FENE_S1 = S/(1 - tr(S)/(2R0^2))
#
# Therefore, the exact STRidge target form is:
#
#   dS/dt = S·kappa + kappa^T·S
#           - (1/lambda) FENE_S
#           + (1/lambda)*beta^-2*d
#
# With the constants used in the FENE-P ground-truth generator:
#
#   1/lambda = 2400/20000 = 0.12
#   (1/lambda)*beta^-2 = 8.0

expected_inv_lambda = 2400.0 / 20000.0
expected_offset = 8.0

print("\n" + "=" * 70)
print("EXPECTED EXACT FENE-P COEFFICIENTS")
print("=" * 70)
print(f"{'d':<20s} {expected_offset:>25.8f}")
print(f"{'UCij':<20s} {1.0:>25.8f}")
print(f"{'LCij':<20s} {0.0:>25.8f}")
print(f"{'S':<20s} {0.0:>25.8f}")
print(f"{'FENE_S1':<20s} {-expected_inv_lambda:>25.8f}")


# -----------------------------
# 11. SAVE RESULTS
# -----------------------------

results_df = pd.DataFrame({
    "term":                 term_names,
    "coefficient_physical": xi_physical,
})

results_df.to_csv(coefficients_csv, index=False)

print("\nSaved coefficients to:")
print(f"  {coefficients_csv}")


# In[ ]:





# In[1]:


# Import numpy for arrays and matrix multiplication
import numpy as np

# Import pandas for the long-format DataFrame and for printing tables
import pandas as pd

# Import PySR's regressor class
from pysr import PySRRegressor


# -----------------------------
# 1. USER SETTINGS
# -----------------------------

# Filenames produced by the FENE-P ground-truth generator
S_npy     = "fenep_merged_S.npy"
dS_npy    = "fenep_merged_dS_dt.npy"
kappa_npy = "fenep_merged_kappa.npy"
labels_npy = "fenep_merged_labels.npy"

# FENE-P denominator constant.
# In the FENE-P ground-truth code, DENOM0 = 2R0^2 = 20000.
DENOM0 = 20000.0

# PySR search budget.
# Since the upper-convected terms are lumped into UC, the search space
# is smaller than when all convected candidates are given separately.
pysr_niterations     = 500
pysr_population_size = 50
pysr_maxsize         = 24

# Output filename for the saved PySR equations CSV
pysr_equations_csv = "pysr_fenep_lumped_UC_discovered_equations.csv"


# -----------------------------
# 2. LOAD FENE-P GROUND-TRUTH TENSOR DATA
# -----------------------------

# FENE-P conformation tensor field, shape (N, 3, 3)
S = np.load(S_npy)

# Time derivative of the conformation tensor, shape (N, 3, 3).
# This is the regression target.
dS_dt = np.load(dS_npy)

# Velocity-gradient tensor, shape (3, 3).
# Constant in time for this simple-shear trajectory.
kappa = np.load(kappa_npy)

# Labels showing which case each sample came from
labels = np.load(labels_npy)

# Number of merged tensor samples
N = S.shape[0]

# Sanity-check the loaded arrays
print(f"Loaded merged data with {N} samples.")
print(f"S shape:       {S.shape}")
print(f"dS_dt shape:   {dS_dt.shape}")
print(f"kappa shape:   {kappa.shape}")
print(f"labels shape:  {labels.shape}")


# Basic checks
if S.shape != dS_dt.shape:
    raise ValueError("S and dS_dt must have the same shape.")

if kappa.shape != S.shape:
    raise ValueError(
        f"For merged data, kappa must have shape {S.shape}, but got {kappa.shape}"
    )

if labels.shape[0] != N:
    raise ValueError(
        f"labels must have length {N}, but got {labels.shape[0]}"
    )

# -----------------------------
# 3. COMPUTE TENSOR FEATURES
# -----------------------------
# For merged data, kappa is already shape (N,3,3).
# So we do not broadcast it.

kappaT = np.transpose(kappa, (0, 2, 1))

# Batch matrix multiplication
S_dot_kappa = S @ kappa
kappaT_dot_S = kappaT @ S

# Lumped upper-convected contribution
UC = S_dot_kappa + kappaT_dot_S

# Trace of S, shape (N,)
tr_S = np.trace(S, axis1=1, axis2=2)


# -----------------------------
# 4. STACK INTO LONG-FORMAT (N × 9 ROWS)
# -----------------------------

# Helper: flatten an (N, 3, 3) tensor field to a (N*9,) column vector
def flat(tensor_field):
    return tensor_field.reshape(-1)

# Build the identity tensor field d = δ_ij.
# This is included because the FENE-P equation has an isotropic source term:
#   (1/lambda)*beta^-2*d
d1 = np.eye(3)
d_field = np.broadcast_to(d1, (N, 3, 3))
factor_d = 8 * d_field

# Flatten the target tensor dS/dt - UC - 8d
y = flat(dS_dt) - flat(UC) - flat(factor_d)

# Convert tr(S), which is naturally shape (N,), into a stacked scalar feature.
# Each time-point trace value is repeated over the 9 tensor components
# so that it aligns with the flattened tensor-component rows.
td = tr_S/DENOM0
tr_S_feature = np.repeat(td, 9)
one_minus_tr_S_feature = 1 - tr_S_feature
F1 = flat(S) / one_minus_tr_S_feature
F2 = (flat(S) * tr_S_feature) / one_minus_tr_S_feature
F3 = (flat(S) * tr_S_feature * tr_S_feature) / one_minus_tr_S_feature
F4 = (flat(S) * tr_S_feature * tr_S_feature * tr_S_feature) / one_minus_tr_S_feature

# Stack the feature columns into a feature matrix.
#
# Feature meanings:
#   Sij           = S_ij
#   tr_Sden          = tr(S)/20000, repeated for each tensor component
#
# We lump the upper-convected terms together as UC so PySR does not
# waste search effort mixing separate convected candidates.
#
# We do NOT give PySR the exact FENE_S term here.
# Instead, PySR receives Sij and tr_S separately and is allowed to use division.
X = np.stack(
    [
        F1,
        F2,
        F3,
        F4
    ],
    axis=1,
)

# Names for the feature columns; PySR uses these in the discovered equation.
# Use Sij instead of S because capital S is reserved internally by SymPy/PySR.
feature_names = [
    "F1",
    "F2",
    "F3",
    "F4"
]

# Sanity-check the shapes
print(f"Feature matrix X shape: {X.shape}")     # expect (N*9, 2)
print(f"Target vector  y shape: {y.shape}")     # expect (N*9,)
print(f"Feature names: {feature_names}\n")


# -----------------------------
# 5. CONFIGURE AND RUN PySR
# -----------------------------
#
# For FENE-P, division is important because the model contains:
#
#   S/(1 - tr(S)/(2R0^2))
#
# Therefore, the operator set includes +, -, *, and /.
# -----------------------------

# Build the PySR regressor.
# parsimony controls how strongly PySR penalizes equation complexity.
model = PySRRegressor(
    niterations         = pysr_niterations,
    population_size     = pysr_population_size,
    maxsize             = pysr_maxsize,
    binary_operators    = ["+", "-", "*"],
    unary_operators     = [],
    elementwise_loss    = "loss(prediction, target) = (prediction - target)^2",
    progress            = False,
    verbosity           = 0,
    random_state        = 42,
    deterministic       = True,
    procs               = 0,
    parallelism         = "serial",
    parsimony           = 0.001,
    batching            = True,
    batch_size          = 10000,
)

# Inform the user that the search is starting
print("Starting PySR search... (first call may take ~30–60 s to precompile)\n")

# Run the regression with readable variable names
model.fit(X, y, variable_names=feature_names)

# Save discovered equations manually after fitting
model.equations_.to_csv(pysr_equations_csv, index=False)


# -----------------------------
# 6. REPORT DISCOVERED EQUATIONS
# -----------------------------

# Print the full Pareto front of equations (complexity vs loss tradeoff)
print("\n" + "=" * 70)
print("PYSR DISCOVERED EQUATIONS (full Pareto front)")
print("=" * 70)
print(model)

# Best equation (PySR's default selection: best loss-per-complexity)
print("\n" + "=" * 70)
print("PYSR'S BEST PICK")
print("=" * 70)
print(f"  Equation: {model.sympy()}")
print(f"  Loss:     {model.get_best()['loss']:.4e}")
print(f"  Complexity: {model.get_best()['complexity']}")


# In[2]:


# Show all PySR equations fully
eqs = model.equations_

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print(eqs[["complexity", "loss", "score", "equation"]].to_string(index=False))


# In[ ]:





# In[10]:


import numpy as np

elong_rates = np.round(np.geomspace(0.005, 0.34, 10), 4)
shear_rates = np.round(np.geomspace(0.005, 1.60, 10), 4)

print(elong_rates)
print(shear_rates)


# In[ ]:





# In[67]:


import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DATA FROM YOUR TABLE
# ============================================================

# -----------------------------
# Elongational flow
# -----------------------------
Wi_elong = np.array([
    0.75,
    1.50,
    3.00,
    6.00,
    12.00,
    24.00
])

Wi_elong_rmse_discovered = np.array([
    17.0584,
    79.2694,
    941.9702,
    826.7974,
    627.0187,
    533.5578
])

Wi_elong_rmse_fenep = np.array([
    2.3704,
    134.3182,
    1142.0290,
    871.3809,
    637.7293,
    537.7596
])


# -----------------------------
# Shear flow
# -----------------------------
Wi_shear = np.array([
    0.75,
    1.50,
    3.00,
    6.00,
    12.00,
    24.00
])

Wi_shear_rmse_discovered = np.array([
    6.5011,
    11.9770,
    31.6383,
    56.4062,
    167.0770,
    886.0672
])

Wi_shear_rmse_fenep = np.array([
    0.4616,
    1.4913,
    9.1675,
    91.5301,
    596.6738,
    1410.3362
])

# -----------------------------
# Mixed flow
# Each pair is: (shear rate, elongational rate)
# -----------------------------
Wi_mixed_shear = np.array([
    0.75,
    1.50,
    3.00,
    6.00,
    12.00,
    24.00
])

Wi_mixed_elong = np.array([
    0.375,
    0.75,
    1.50,
    3.00,
    6.00,
    12.00
])

Wi_mixed_rmse_discovered = np.array([
    9.3680,
    27.7791,
    64.5865,
    967.1119,
    830.9956,
    626.6571
])

Wi_mixed_rmse_fenep = np.array([
    0.6809,
    6.0481,
    355.4134,
    1149.0687,
    869.2132,
    635.1347
])



# ============================================================
# 2. PLOT ELONGATIONAL FLOW RMSE
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    Wi_elong,
    Wi_elong_rmse_discovered,
    marker="o",
    linewidth=2.2,
    label="Discovered FENE"
)

plt.plot(
    Wi_elong,
    Wi_elong_rmse_fenep,
    marker="s",
    linewidth=2.2,
    linestyle="--",
    label="FENE-P"
)

plt.xlabel("Elongational flow rate")
plt.ylabel("Net RMSE from Exact FENE")
plt.title("Net RMSE from Exact FENE vs Elongational Flow Rate")
plt.grid(True, alpha=0.3)
plt.legend()

# Log scale helps because RMSE changes from small values to very large values
plt.xscale("log")
#plt.yscale("log")

plt.tight_layout()
plt.savefig("net_rmse_Wi_elongational.pdf", bbox_inches="tight")
plt.show()


# ============================================================
# 3. PLOT SHEAR FLOW RMSE
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    Wi_shear,
    Wi_shear_rmse_discovered,
    marker="o",
    linewidth=2.2,
    label="Discovered FENE"
)

plt.plot(
    Wi_shear,
    Wi_shear_rmse_fenep,
    marker="s",
    linewidth=2.2,
    linestyle="--",
    label="FENE-P"
)

plt.xlabel("Shear flow rate")
plt.ylabel("Net RMSE from Exact FENE")
plt.title("Net RMSE from Exact FENE vs Shear Flow Rate")
plt.grid(True, alpha=0.3)
plt.legend()

# Log scale helps because both flow rate and RMSE span wide ranges
plt.xscale("log")
#plt.yscale("log")

plt.tight_layout()
plt.savefig("net_rmse_Wi_shear.pdf", bbox_inches="tight")
plt.show()


# ============================================================
# 4. PLOT MIXED FLOW RMSE
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    Wi_mixed_shear,
    Wi_mixed_rmse_discovered,
    marker="o",
    linewidth=2.2,
    label="Discovered FENE"
)

plt.plot(
    Wi_mixed_shear,
    Wi_mixed_rmse_fenep,
    marker="s",
    linewidth=2.2,
    linestyle="--",
    label="FENE-P"
)

# Tag each point with its corresponding elongational Wi
# for gamma, eps, yval in zip(Wi_mixed_shear, Wi_mixed_elong, Wi_mixed_rmse_discovered):
#     plt.annotate(
#         rf"$Wi_{{ext}}={eps:.3f}$",
#         xy=(gamma, yval),
#         xytext=(5, 5),
#         textcoords="offset points",
#         fontsize=8
#     )

plt.xlabel(r"Shear Weissenberg Number in mixed flow, $Wi_{\mathrm{shear}}$")
plt.ylabel("Net RMSE from Exact FENE")
plt.title("Net RMSE from Exact FENE under Mixed Flow")
plt.grid(True, alpha=0.3)
plt.legend()

plt.xscale("log")
# plt.yscale("log")

plt.tight_layout()
plt.savefig("net_rmse_mixed_flow_vs_shear_Wi.pdf", bbox_inches="tight")
plt.show()


# In[115]:


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Define raw data from mixed flow planar extensional & shear tests
# ----------------------------------------------------------------------
Wi_mixed_shear = np.array([0.75, 1.50, 3.00, 6.00, 12.00, 24.00])
Wi_mixed_ext = np.array([0.375, 0.75, 1.50, 3.00, 6.00, 12.00])

Wi_mixed_rmse_discovered = np.array([14.4991, 18.4247, 6.4466, 22.9417, 13.2181, 8.6384])
Wi_mixed_rmse_fenep = np.array([1.0710, 4.1358, 35.9257, 27.2673, 13.8231, 8.7566])

# ----------------------------------------------------------------------
# 2. Compute the Effective Weissenberg Number (Wi_eff)
#    Derived for mixed simple shear and planar extensional flow:
#    Wi_eff = sqrt(Wi_shear^2 + 4 * Wi_elong^2)
# ----------------------------------------------------------------------
Wi_eff = np.sqrt(Wi_mixed_shear**2 + 4.0 * Wi_mixed_ext**2)

# ----------------------------------------------------------------------
# 3. Plotting Setup (Publication Quality)
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 19,
    'axes.labelsize': 20,
    'axes.titlesize': 21,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 19,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, ax1 = plt.subplots(figsize=(10.5, 8.1), dpi=300)


# Plot Discovered Model vs FENE-P Model
ax1.plot(Wi_eff, Wi_mixed_rmse_discovered, 
         marker='o', markersize=8, color='#de425b', linewidth=2, 
         linestyle='-', label='Discovered Model (STRigde-CE)')
ax1.plot(Wi_eff, Wi_mixed_rmse_fenep, 
         marker='s', markersize=8, color='#004c6d', linewidth=2, 
         linestyle='--', label='Analytical FENE-P Model')
# Red RMSE labels below the red points
#for x, y in zip(Wi_eff, Wi_mixed_rmse_discovered): ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, -9), textcoords='offset points', ha='center', va='top', fontsize=8, color='#de425b')

# Blue RMSE labels above the blue points
#for x, y in zip(Wi_eff, Wi_mixed_rmse_fenep): ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='#004c6d')

# Adjust scale & labels for primary bottom axis
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xticks(Wi_eff)
ax1.set_xticklabels([f'{x:.1f}' for x in Wi_eff])
ax1.set_xlabel(r'Effective Weissenberg Number, $Wi_{\mathrm{eff}} = \sqrt{Wi_{\mathrm{shear}}^2 + 4Wi_{\mathrm{ext}}^2}$', labelpad=10)
ax1.set_ylabel('Relative Error from Exact FENE (%)', labelpad=10)
#ax1.set_title('Constitutive Model Net RMSE under Planar Mixed Flow', pad=25, fontweight='bold')
ax1.grid(True, which='both')

# Set aesthetic limits
ax1.set_xlim(0.8, 40)
ax1.set_ylim(0.0, 60)

# Add Legend
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', shadow=True)

# ----------------------------------------------------------------------
# 4. Add Secondary Top X-Axis for (Wi_shear, Wi_ext) Coordinate Labels
# ----------------------------------------------------------------------
ax2 = ax1.twiny()
ax2.set_xscale('log')
ax2.set_xlim(ax1.get_xlim())

# Align ticks of top axis exactly with the calculated effective Wi data points
ax2.set_xticks(Wi_eff)

# Create tick labels showing the mixed coordinate tuples: (Wi_shear, Wi_elong)
labels = [
    f"({s:.2f}, {e:.3f})" if i == 0 else f"({s:.2f}, {e:.2f})"
    for i, (s, e) in enumerate(zip(Wi_mixed_shear, Wi_mixed_ext))
]
ax2.set_xticklabels(labels, rotation=45, ha='left', fontsize=18)
ax2.set_xlabel(r'Constituent Coordinates $(Wi_{\mathrm{shear}},\ Wi_{\mathrm{ext}})$', labelpad=10)

plt.tight_layout()

# Save and show the plot
plt.savefig('mixed_flow_rmse_analysis.png', bbox_inches='tight')
plt.show()


# In[116]:


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Define raw data from pure elongational flow tests
# ----------------------------------------------------------------------
Wi_ext = np.array([0.75, 1.5, 3.0, 6.0, 12.0, 24.0])

Wi_ext_rmse_discovered = np.array([17.5784, 13.0286, 24.4628, 13.4233, 8.7162, 7.2877])
Wi_ext_rmse_fenep = np.array([2.3025, 21.0243, 29.6771, 14.1464, 8.8652, 7.3394])

# ----------------------------------------------------------------------
# 2. Plotting Setup (Publication Quality)
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 19,
    'axes.labelsize': 20,
    'axes.titlesize': 21,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 19,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, ax1 = plt.subplots(figsize=(8.5, 6.3), dpi=300)

# Plot Discovered Model vs FENE-P Model
ax1.plot(Wi_ext, Wi_ext_rmse_discovered, 
         marker='o', markersize=8, color='#de425b', linewidth=2, 
         linestyle='-', label='Discovered Model (STRidge-CE)')
ax1.plot(Wi_ext, Wi_ext_rmse_fenep, 
         marker='s', markersize=8, color='#004c6d', linewidth=2, 
         linestyle='--', label='Analytical FENE-P Model')

# Red RMSE labels below the red points
#for x, y in zip(Wi_ext, Wi_ext_rmse_discovered): ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, -13), textcoords='offset points', ha='center', va='top', fontsize=8, color='#de425b')

# Blue RMSE labels above the blue points
#for x, y in zip(Wi_ext, Wi_ext_rmse_fenep): ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='#004c6d')

# Adjust scale & labels for primary axis
# Keep the x-axis linear, and y-axis log due to large orders of magnitude of RMSE
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel(r'Extensional Weissenberg Number, $Wi_{\mathrm{ext}}$', labelpad=10)
ax1.set_ylabel('Relative Error from Exact FENE (%)', labelpad=10)
#ax1.set_title('Constitutive Model Net RMSE under Pure Extensional Flow', pad=15, fontweight='bold')
ax1.grid(True, which='both')

# Set aesthetic limits
ax1.set_xlim(0.6, 30)
ax1.set_ylim(0, 60)

# Set ticks on the x-axis explicitly to match your data points
ax1.set_xticks(Wi_ext)
ax1.set_xticklabels(['0.75', '1.5', '3.0', '6.0', '12.0', '24.0'], rotation=45, ha='right', fontsize=18)

# Add Legend
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', shadow=True)

plt.tight_layout()

# Save and show the plot
plt.savefig('extensional_flow_rmse_analysis.png', bbox_inches='tight')
plt.show()


# In[117]:


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Define raw data from pure shear flow tests
# ----------------------------------------------------------------------
Wi_shear = np.array([0.75, 1.50, 3.00, 6.00, 12.00, 24.00])

Wi_shear_rmse_discovered = np.array([12.2557, 15.8628, 17.7091, 10.3092, 12.2828, 37.2595])
Wi_shear_rmse_fenep = np.array([0.8036, 1.4202, 4.3956, 16.8903, 44.1253, 59.2987])

# ----------------------------------------------------------------------
# 2. Plotting Setup (Publication Quality)
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 19,
    'axes.labelsize': 20,
    'axes.titlesize': 21,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 19,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, ax1 = plt.subplots(figsize=(8.5, 6.3), dpi=300)

# Plot Discovered Model vs FENE-P Model
ax1.plot(Wi_shear, Wi_shear_rmse_discovered, 
         marker='o', markersize=8, color='#de425b', linewidth=2, 
         linestyle='-', label='Discovered Model (STRidge-CE)')
ax1.plot(Wi_shear, Wi_shear_rmse_fenep, 
         marker='s', markersize=8, color='#004c6d', linewidth=2, 
         linestyle='--', label='Analytical FENE-P Model')

# Red RMSE labels below the red points
#for x, y in zip(Wi_ext, Wi_shear_rmse_discovered): ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=8, color='#de425b')

# Blue RMSE labels above the blue points
#for x, y in zip(Wi_ext, Wi_shear_rmse_fenep): ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='#004c6d')

# Adjust scale & labels for primary axis
# Keep the x-axis linear, and y-axis log due to large orders of magnitude of RMSE
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel(r'Shear Weissenberg Number, $Wi_{\mathrm{shear}}$', labelpad=10)
ax1.set_ylabel('Relative Error from Exact FENE (%)', labelpad=10)
#ax1.set_title('Constitutive Model Net RMSE under Pure Shear Flow', pad=15, fontweight='bold')
ax1.grid(True, which='both')

# Set aesthetic limits
ax1.set_xlim(0.6, 30)
ax1.set_ylim(0.0, 60)

# Set ticks on the x-axis explicitly to match your data points
ax1.set_xticks(Wi_shear)
ax1.set_xticklabels(['0.75', '1.5', '3.0', '6.0', '12.0', '24.0'], rotation=45, ha='right', fontsize=18)

# Add Legend
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', shadow=True)

plt.tight_layout()

# Save and show the plot
plt.savefig('shear_flow_rmse_analysis.png', bbox_inches='tight')
plt.show()


# In[ ]:




