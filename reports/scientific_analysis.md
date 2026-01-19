# Scientific Code Analysis and Methodology Reproduction Verification

## 1. Executive Summary

This report analyzes three implementations of the "Equivariant Transition Matrices" manuscript:

1. **Main Project** (`d:\GitHub\transition-matrix-equiv`)
2. **Etm Repro** (`SUB_PROJECTS\etm_xai_repro`)
3. **K-Web** (`SUB_PROJECTS\kweb`)

All three projects attempt to reproduce the methodology defined in `PROMPT.md`. The analysis focuses on compliance with the manuscript's requirements, scientific validity of the implementations, and a comparative assessment of their strengths and weaknesses.

**Key Finding:** `kweb` is the most scientifically rigorous and complete implementation, explicitly documenting negative results that challenge the naive application of the methodology to deep neural networks. `Main Project` and `etm_xai_repro` provide solid core implementations but lack the extensive experimental scaffolding and critical self-analysis found in `kweb`.

## 2. Methodology Reproduction Compliance

The manuscript in `PROMPT.md` mandates several key components. Here is the compliance matrix:

| Requirement | Main Project | Etm Repro | K-Web |
| :--- | :---: | :---: | :---: |
| **A. Baseline (Fidelity-only)** | ✅ Implemented | ✅ Implemented | ✅ Implemented (`02_baseline`) |
| **B. Equivariant Solver (Algorithm 1)** | ✅ Implemented (LSQR/SVD) | ✅ Implemented (CG/SVD) | ✅ Implemented (LSQR/CG) |
| **C. Generator Estimation (Finite Diff)** | ✅ Implemented | ✅ Implemented | ✅ Implemented |
| **D. Synthetic Experiment (Bridge)** | ✅ Implemented | ✅ Implemented | ✅ Implemented (`04_synthetic`) |
| **E. MNIST Experiment (Robustness)** | ⚠️ Partial (Script exists) | ⚠️ Partial | ✅ Full (`05_mnist_*`) |
| **F. Scalable Solver (LinearOperator)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **G. Visualizations** | ⚠️ Basic | ⚠️ Basic | ✅ Comprehensive |
| **H. Scientific Critique/Self-Eval** | ❌ Minimal | ❌ Minimal | ✅ Extensive |

### Detailed Breakdown

* **Main Project**: Contains the core logic in `src/equivariant.py` and `src/generators.py`. It correctly implements the iterative solver using `LinearOperator` to avoid Kronecker products. However, it appears to be a library-focused implementation rather than a full reproduction pipeline. The experimental scripts are minimal.
* **Etm Repro**: Similar to Main, it provides a library structure (`src/methods`). It uses Conjugate Gradient (CG) on normal equations for the solver, which is a valid alternative to LSQR. It includes `ridge` regularization in generator estimation, which is a rigorous detail.
* **K-Web**: This is a full research workflow. It includes numbered scripts (`01` to `07`) that sequentially build up the reproduction. It explicitly handles the "Bridge" method for synthetic data and conducts the full MNIST robustness test, producing detailed metrics and plots.

## 3. Comparative Code Analysis

### 3.1 Equivariant Solver Implementation

The core mathematical problem is solving:
$$ \min_T \|B^\top - T A^\top\|_F^2 + \lambda \|T J^A - J^B T\|_F^2 $$

* **Main Project (`compute_equivariant_T_iterative`)**:
  * Uses `scipy.sparse.linalg.lsqr`.
  * Defines `matvec` and `rmatvec` for the combined system matrix.
  * Correctly handles the block structure: Fidelity block + Symmetry block.
  * **Verdict**: mathematically correct and scalable.

* **Etm Repro (`solve_equivariant_large_cg`)**:
  * Uses `scipy.sparse.linalg.cg`.
  * Solves the **normal equations** implicitly: $(A^\top A \otimes I + \dots)\text{vec}(T) = \dots$
  * Includes `ridge` (Tikhonov) regularization support.
  * **Verdict**: Mathematically equivalent (if conditioned well), but solving normal equations can sometimes be less numerically stable than LSQR on the original system.

* **K-Web (`EquivariantSolver`)**:
  * Supports both `lsqr` and `cg`.
  * Implementation in `solver.py` is very readable and explicitly comments the shapes and block structures.
  * Calculates detailed breakdown of objective components (reconstruction vs symmetry).
  * **Verdict**: Most flexible and well-instrumented.

### 3.2 Generator Estimation

The manuscript requires estimating generators $J$ via:
$$ \Delta A \approx A J^\top $$

* **Main & Etm Repro**: Both solve this using least squares (`lstsq` or `pinv`). `Etm Repro` adds ridge regularization `_ridge_solve`, which helps when features are collinear (common in CNNs).
* **K-Web**: Uses `lstsq` and explicitly checks and logs the **antisymmetry** of the resulting matrices in its experiments. This is a critical scientific step, as true rotation generators must be antisymmetric. K-Web's logs reveal that for MNIST, the estimated matrices are **not** antisymmetric, invalidating the theoretical assumption – a crucial negative finding.

## 4. Critical Scientific Analysis

The "Bridge" method for synthetic data is implemented in all three, using MDS + Linear Regression. This essentially forces a linear structure, which explains why the baseline model works so well in the synthetic experiments (as noted in K-Web's results).

**The Elephant in the Room: Nonlinearity**
All implementations face the same fundamental scientific hurdle described in the manuscript: applying linear Lie algebra theory to nonlinear deep learning features.

* **Theory**: $f(g \cdot x) \approx \rho(g) f(x)$. This implies the feature map $f$ is equivariant.
* **Reality**: CNNs are not naturally equivariant to continuous rotations (unless specialized like G-CNNs).
* **Consequence**: The "generators" estimated via finite differences on CNN features are not true Lie algebra generators. They are just local linear approximations of the network's sensitivity to rotation.
* **Outcome**: The symmetry constraint $\|T J^A - J^B T\|$ forces $T$ to respect a structure that doesn't actually exist globally in the data.

**K-Web's contribution** is identifying this. It explicitly reports that the equivariant method performs **worse** than the baseline on MNIST (SSIM 0.007 vs 0.168). This is a scientifically valid negative result. The other projects provide the *tools* to find this, but do not emphasize the result itself.

## 5. Pros and Cons

### Main Project

* **Pros**: Clean, standard Python package structure. Good separation of concerns (`core`, `equivariant`, `generators`).
* **Cons**: Lacks the narrative "reproducibility" scripts. User has to figure out how to wire components together for an experiment.

### Etm Repro

* **Pros**: Good mathematical hygiene (ridge regularization, CG solver). Type hinting is solid.
* **Cons**: Like Main, it is a library, not a "lab notebook". Lacks the explicit narrative of the experiment execution.

### K-Web

* **Pros**:
  * **Complete Workflow**: Scripts `01`–`07` guide the user through the entire scientific process.
  * **Transparency**: Logs and READMEs explicitly detail the *failure* of the method on MNIST, which is high-integrity science.
  * **Verification**: Includes self-verification steps (`03_methodology_test`) ensuring the solver works on ideal data before trying real data.
* **Cons**: The "workflow" structure is distinct from a standard PyPI package structure (though it has `src` equivalent in `workflow`).

## 6. Conclusion and Recommendation

**K-Web** is the superior project in terms of **scientific reproducibility**. It not only implements the algorithms but also rigorously tests the hypothesis, leading to a properly documented negative result for the MNIST experiment.

If the goal is to **use the library** in another project:

* Adopt the **Main Project** structure but integrate **Etm Repro's** ridge regularization.

If the goal is to **verify the manuscript**:

* **K-Web** has already done it. It proves that while the math works (synthetic data), the application to standard CNNs for rotations is flawed due to the lack of native equivariance in the features. The methodology requires equivariant architectures (G-CNNs) to work as intended.
