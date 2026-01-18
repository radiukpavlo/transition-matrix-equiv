# Reproducibility Checklist

This document provides all information needed to reproduce the experiments in
"Equivariant Transition Matrices for Explainable Deep Learning: A Lie Group Linearization Approach".

## Environment

### Software Versions

| Component | Version | Installation |
|-----------|---------|--------------|
| Python | 3.10+ | Required |
| NumPy | 1.24+ | `pip install numpy` |
| SciPy | 1.10+ | `pip install scipy` |
| scikit-learn | 1.3+ | `pip install scikit-learn` |
| Matplotlib | 3.7+ | `pip install matplotlib` |
| PyTorch | 2.0+ | (MNIST experiment only) `pip install torch torchvision` |
| pytest | 8.0+ | (testing only) `pip install pytest` |

### Install All Dependencies

```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **CPU**: Any modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended for MNIST experiments
- **GPU**: Optional, accelerates CNN training for MNIST

## Random Seeds

All experiments use fixed seeds for reproducibility:

| Experiment | Seed | Usage |
|------------|------|-------|
| Synthetic | 42 | MDS, random rotations |
| MNIST | 42 | Data splits, weight initialization |
| Generator estimation | 42 | MDS bridge fitting |

## Data Sources

### Synthetic Data (Section 3.4)

- Matrices A (15×5) and B (15×4) are from manuscript Appendix 1.1
- Hardcoded in `experiments/synthetic/data.py`
- No external data files needed

### MNIST Dataset (Section 3.5)

- Standard MNIST dataset from torchvision
- Automatically downloaded on first run
- 60,000 training + 10,000 test images

## Experiment Commands

### Run All Tests

```bash
cd d:\GitHub\transition-matrix-equiv
python -m pytest tests/ -v
```

### Synthetic Experiment (Section 3.4)

```bash
python experiments/synthetic/run_experiment.py --validate
```

**Expected output (approximate):**

- MSE_old ≈ 0.003-0.004
- Sym_err_old ≈ 13000+ (varies with MDS)
- Sym_err_new ≈ 0.04 (much lower with equivariant approach)

### Generate Figures

```bash
python scripts/generate_figures.py --output figures/
```

**Generated files:**

- `figures/fig1_mds_visualization.png` - MDS scatter of A and B
- `figures/fig2_robustness_scatter.png` - Editor's requested comparison
- `figures/fig3_lambda_tradeoff.png` - Fidelity vs symmetry defect

### MNIST Experiment (Section 3.5)

```bash
python experiments/mnist/run_experiment.py
```

**Note:** Requires PyTorch and may take 10-30 minutes depending on hardware.

## Hyperparameters

### Generator Estimation

| Parameter | Value | Description |
|-----------|-------|-------------|
| ε (epsilon) | 0.01 rad | Small rotation angle for finite differences |
| MDS components | 2 | Reduction dimension for synthetic bridge |

### Equivariant Solver

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ (lambda) | 0.5 | Default balance between fidelity and equivariance |
| τ (tau) | 1e-10 | SVD regularization threshold |
| r | 1 | Number of generators (SO(2) rotation) |

### MNIST CNN Architecture

| Layer | Output Shape | Details |
|-------|--------------|---------|
| Conv2d | 32×26×26 | 3×3 kernel, ReLU |
| Conv2d | 64×24×24 | 3×3 kernel, ReLU |
| MaxPool2d | 64×12×12 | 2×2 pool |
| Dropout | - | p=0.25 |
| Flatten | 9216 | - |
| Linear | 490 | ReLU (FM features) |
| Dropout | - | p=0.5 |
| Linear | 10 | Softmax |

## Directory Structure

```
transition-matrix-equiv/
├── src/                     # Core library
│   ├── core.py             # vec, devec, SVD solver
│   ├── baseline.py         # Old approach (T_old)
│   ├── equivariant.py      # New approach (T_new)
│   ├── generators.py       # Lie algebra estimation
│   └── metrics.py          # MSE, symmetry defect, SSIM, PSNR
├── experiments/
│   ├── synthetic/          # Section 3.4
│   │   ├── data.py         # Matrices A, B from appendix
│   │   └── run_experiment.py
│   └── mnist/              # Section 3.5
│       ├── model.py        # CNN architecture
│       ├── features.py     # Feature extraction
│       └── run_experiment.py
├── scripts/
│   └── generate_figures.py # All visualizations
├── tests/                  # Unit tests
├── figures/                # Generated output
└── reports/               # Documentation
```

## Known Issues and Workarounds

1. **MDS Warning**: scikit-learn 1.9 will change `n_init` default. This doesn't affect results.

2. **Numerical Differences**: Results may vary slightly (~1-5%) across different NumPy/BLAS configurations due to floating-point arithmetic ordering.

3. **Memory for MNIST**: If memory is limited, use `compute_equivariant_T_iterative()` instead of the explicit solver.

## Validation Checks

Run these to verify installation:

```python
# Quick sanity check
import numpy as np
from src.core import vec, devec
T = np.random.randn(4, 5)
assert np.allclose(T, devec(vec(T), (4, 5)))
print("✓ Core operations working")

# Test baseline
from experiments.synthetic.data import get_matrix_A, get_matrix_B
from src.baseline import compute_baseline_T
A, B = get_matrix_A(), get_matrix_B()
T_old = compute_baseline_T(A, B)
assert T_old.shape == (4, 5)
print("✓ Baseline solver working")
```

## Contact

For questions about reproducing results, please open an issue on the repository.
