# Equivariant Transition Matrices Research Reproduction Plan

Reproduce the methodology from the manuscript "Equivariant Transition Matrices for Explainable Deep Learning: A Lie Group Linearization Approach" with complete computational experiments.

## User Review Required

> [!IMPORTANT]
> **PDF Baseline Reference**: The `previous_approach.pdf` cannot be programmatically parsed. The plan uses methodology details from the manuscript's references to [4] and the equations in the prompt. Please confirm this approach.

> [!WARNING]
> **MNIST Experiment Scale**: The full MNIST experiment with k=490, ℓ=784 creates large matrices. The plan includes memory-efficient iterative solvers. GPU acceleration recommended but not required.

---

## Proposed Changes

### Core Library Module

Creates the foundational mathematical framework.

#### [NEW] [src/\_\_init\_\_.py](file:///d:/GitHub/transition-matrix-equiv/src/__init__.py)

Package initialization.

#### [NEW] [src/core.py](file:///d:/GitHub/transition-matrix-equiv/src/core.py)

Core mathematical operations:

- `vec(T)`: Vectorization operation for matrices
- `devec(u, shape)`: De-vectorization (reshape)
- `kron_mvp(A, B, x)`: Memory-efficient Kronecker-vector product without explicit formation
- `svd_solve(M, Y, tau)`: SVD-based pseudoinverse solver with regularization

#### [NEW] [src/baseline.py](file:///d:/GitHub/transition-matrix-equiv/src/baseline.py)

Baseline transition matrix (Old Approach from [4]):

- `compute_baseline_T(A, B)`: Compute T_old = pinv(A) @ B via SVD
- Uses fidelity-only objective: min ||B - AT^⊤||²_F

#### [NEW] [src/generators.py](file:///d:/GitHub/transition-matrix-equiv/src/generators.py)

Lie algebra generator estimation:

- `estimate_generator_synthetic(A, A_rot, epsilon)`: For synthetic data via MDS bridge
- `estimate_generator_features(features, features_rot, epsilon)`: For real feature matrices
- `rotate_2d_points(points, angle)`: SO(2) rotation helper

#### [NEW] [src/equivariant.py](file:///d:/GitHub/transition-matrix-equiv/src/equivariant.py)

Equivariant transition matrix (New Approach):

- `build_fidelity_system(A, B)`: Form M_fid and Y_fid
- `build_symmetry_system(J_A, J_B, k, l)`: Form K_i constraint matrices
- `compute_equivariant_T(A, B, J_A, J_B, lambda_, tau)`: Full Algorithm 1 implementation
- `compute_equivariant_T_iterative(...)`: Memory-efficient version for large matrices

#### [NEW] [src/metrics.py](file:///d:/GitHub/transition-matrix-equiv/src/metrics.py)

Evaluation metrics:

- `fidelity_mse(B, B_pred)`: MSE between original and predicted features
- `symmetry_defect(T, J_A, J_B)`: ||TJ^A - J^BT||²_F
- `compute_ssim(img1, img2)`: Structural Similarity Index
- `compute_psnr(img1, img2)`: Peak Signal-to-Noise Ratio

---

### Synthetic Experiment Module

Implements Section 3.4 with all three scenarios.

#### [NEW] [experiments/synthetic/data.py](file:///d:/GitHub/transition-matrix-equiv/experiments/synthetic/data.py)

Synthetic data from manuscript Appendix 1.1:

- Matrices A (15×5), B (15×4)
- Class labels for visualization

#### [NEW] [experiments/synthetic/algorithm2.py](file:///d:/GitHub/transition-matrix-equiv/experiments/synthetic/algorithm2.py)

Algorithm 2 implementation (MDS + decoder bridge):

- MDS reduction to 2D
- Linear regression decoder training
- Rotation and inverse mapping

#### [NEW] [experiments/synthetic/run_experiment.py](file:///d:/GitHub/transition-matrix-equiv/experiments/synthetic/run_experiment.py)

Complete synthetic experiment:

- Scenario 1: Baseline T_old, compute MSE and Sym_err
- Scenario 2: Equivariant T_new with λ=0.5
- Scenario 3: Robustness test with random rotations ±15°
- Table 1 generation
- **Scatter plot figures** (Editor's requirement)

---

### MNIST Experiment Module

Implements Section 3.5.

#### [NEW] [experiments/mnist/model.py](file:///d:/GitHub/transition-matrix-equiv/experiments/mnist/model.py)

CNN architecture:

- Similar to [4]: Conv layers → FC layers → k=490 features
- Feature extraction from penultimate layer

#### [NEW] [experiments/mnist/features.py](file:///d:/GitHub/transition-matrix-equiv/experiments/mnist/features.py)

Feature extraction:

- FM features: CNN penultimate layer (k=490)
- MM features: Flattened pixels (ℓ=784)
- Rotation augmentation for generator estimation

#### [NEW] [experiments/mnist/run_experiment.py](file:///d:/GitHub/transition-matrix-equiv/experiments/mnist/run_experiment.py)

MNIST experiment pipeline:

- Train CNN, extract features
- Estimate generators J^A, J^B
- Compute T_old and T_new
- Evaluate reconstruction (SSIM, PSNR)
- Test robustness on rotated test set

---

### Visualization & Reports

#### [NEW] [scripts/generate_figures.py](file:///d:/GitHub/transition-matrix-equiv/scripts/generate_figures.py)

All required figures:

- **Fig. 2**: Side-by-side scatter plots for robustness test
  - Left: B*_old_rot (chaotic)
  - Right: B*_new_rot (structured)
- **Fig. 3**: MNIST reconstructions comparison
- **Fig. 4**: λ trade-off curve (fidelity vs symmetry defect)
- **Fig. 5**: SSIM/PSNR distributions

#### [NEW] [reports/reproduction_report.md](file:///d:/GitHub/transition-matrix-equiv/reports/reproduction_report.md)

Structured report with 10 sections per prompt requirements.

#### [NEW] [reports/reproducibility_checklist.md](file:///d:/GitHub/transition-matrix-equiv/reports/reproducibility_checklist.md)

Seeds, versions, hardware notes, exact commands.

---

### Tests

#### [NEW] [tests/test_core.py](file:///d:/GitHub/transition-matrix-equiv/tests/test_core.py)

Unit tests for algebraic identities:

- `test_vec_devec_identity()`: vec(devec(u)) == u
- `test_kronecker_vec_identity()`: vec(AXB) == (B^T ⊗ A)vec(X)
- `test_svd_solve_accuracy()`: SVD solver correctness

#### [NEW] [tests/test_generators.py](file:///d:/GitHub/transition-matrix-equiv/tests/test_generators.py)

Generator estimation tests:

- `test_so2_generator_structure()`: J should be antisymmetric
- `test_small_epsilon_stability()`: Generator stable for small ε

#### [NEW] [tests/test_equivariant.py](file:///d:/GitHub/transition-matrix-equiv/tests/test_equivariant.py)

Equivariant solver tests:

- `test_lambda_zero_equals_baseline()`: λ=0 gives T_old
- `test_symmetry_defect_decreases()`: Higher λ → lower symmetry defect

---

## Verification Plan

### Automated Tests

**Run all unit tests:**

```bash
cd d:\GitHub\transition-matrix-equiv
python -m pytest tests/ -v --tb=short
```

**Run synthetic experiment with assertions:**

```bash
python experiments/synthetic/run_experiment.py --validate
```

Expected outputs:

- MSE_old ≈ 0.002, Sym_err_old ≈ 1.450
- MSE_new ≈ 0.005, Sym_err_new ≈ 0.080
- Robustness error: old ≈ 0.850, new ≈ 0.120

### Visual Verification

**Generate and inspect figures:**

```bash
python scripts/generate_figures.py --output figures/
```

- Verify scatter plots show "chaos vs order" pattern
- Verify reconstructed MNIST digits are recognizable

### Manual Verification

1. **Review generated scatter plots in `figures/`** - confirm left plot (old approach) shows mixed/scattered points while right plot (new approach) preserves cluster structure
2. **Review MNIST reconstructions** - confirm digits are visually similar to originals
3. **Check λ trade-off curve** - should show inverse relationship between fidelity and symmetry defect

---

## Memory-Efficient Solver for MNIST-Scale Problems

### The Memory Problem

The equivariant solver requires forming and solving **Mu = Y** where M is constructed from Kronecker products:

```
M = [ A ⊗ I_ℓ        ]   ← Fidelity block: (m·ℓ) × (k·ℓ)
    [ λ · K_1        ]   ← Symmetry block: (k·ℓ) × (k·ℓ)
    
where K_i = (J_i^A)^T ⊗ I_ℓ - I_k ⊗ J_i^B
```

**MNIST dimensions** (per manuscript Section 3.5):

- k = 490 (FM features from CNN penultimate layer)
- ℓ = 784 (MM features = 28×28 pixels)
- m = 1000 (sample size)
- r = 1 (one SO(2) generator)

| Block | Dimensions | Elements | Memory (float64) |
|-------|------------|----------|------------------|
| M_fid = A ⊗ I_ℓ | 784,000 × 384,160 | 301 billion | **2.4 TB** |
| K_1 | 384,160 × 384,160 | 148 billion | **1.2 TB** |

**Total naive storage**: ~3.6 TB — clearly infeasible.

### The Solution: Implicit Linear Operators

We never form M explicitly. Instead, we use `scipy.sparse.linalg.LinearOperator` to define **matrix-vector products** implicitly:

**Key identity**: For Kronecker products, `(A ⊗ B) @ vec(X) = vec(B @ X @ A^T)`

This transforms O(n⁴) storage into O(n²) compute per iteration:

```python
def matvec_fidelity(u, A, l):
    """Compute (A ⊗ I_ℓ) @ u without forming the Kronecker product."""
    T = u.reshape(l, k)           # devec: (k·ℓ,) → (ℓ, k)
    result = T @ A.T              # (ℓ, k) @ (k, m) = (ℓ, m)
    return result.T.ravel()       # (m, ℓ) → (m·ℓ,)

def matvec_symmetry(u, J_A, J_B, k, l):
    """Compute ((J_A)^T ⊗ I_ℓ - I_k ⊗ J_B) @ u without forming K."""
    T = u.reshape(l, k)
    term1 = T @ J_A.T             # (ℓ, k) @ (k, k) = (ℓ, k)
    term2 = J_B @ T               # (ℓ, ℓ) @ (ℓ, k) = (ℓ, k)
    return (term1 - term2).ravel()
```

**Memory reduction**: From **3.6 TB** → **~6 MB** (just storing A, B, J_A, J_B, T)

### Iterative Solver Choice

We use **LSQR** (iterative least-squares) from `scipy.sparse.linalg`:

```python
from scipy.sparse.linalg import LinearOperator, lsqr

# Define implicit operator
M_op = LinearOperator(
    shape=(m*l + k*l, k*l),
    matvec=lambda u: combined_matvec(u, A, J_A, J_B, k, l, lambda_)
)

# Solve iteratively
solution, *info = lsqr(M_op, Y, atol=1e-8, btol=1e-8, iter_lim=500)
T = solution.reshape(l, k)
```

**Why LSQR over Conjugate Gradient?**

- LSQR handles rectangular, overdetermined systems directly
- No need to form M^T M (which would square the condition number)
- Built-in regularization via iteration count

### Computational Summary

| Experiment | Approach | Memory | Time (est.) |
|------------|----------|--------|-------------|
| Synthetic (m=15) | Explicit SVD | 12 KB | < 1 sec |
| MNIST subset (m=500) | Explicit SVD | ~500 MB | ~30 sec |
| MNIST full (m=1000+) | **LSQR + implicit** | ~50 MB | ~5-10 min |

### Mathematical Equivalence Proof

The implicit solver minimizes the **exact same objective**:

$$\mathcal{L}(T) = \|B^\top - T A^\top\|_F^2 + \lambda \sum_{i=1}^r \|T J_i^A - J_i^B T\|_F^2$$

**Proof sketch**: LSQR minimizes ||Mu - Y||₂². By construction:

- First m·ℓ components of Mu - Y equal vec(B^T - TA^T), so squared norm = fidelity term
- Next k·ℓ components equal λ·vec(TJ^A - J^BT), so squared norm = λ² × symmetry term

The solution is mathematically identical to the direct SVD approach, just computed without materializing the full matrix.
