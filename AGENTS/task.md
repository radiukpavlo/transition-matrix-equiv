# K-Dense Web Research Reproduction Task

## Phase 1: Planning & Setup

- [x] Read and understand PROMPT.md requirements
- [x] Create implementation plan
- [x] Request user approval for plan

## Phase 2: Core Implementation

- [x] Create project structure (src/, tests/, figures/, results/)
- [x] Implement core utilities (vectorization, Kronecker products, SVD solver)
- [x] Implement baseline transition matrix (Old Approach)
- [x] Implement generator estimation (Lie algebra)
- [x] Implement equivariant transition matrix (New Approach)
- [x] Implement evaluation metrics (MSE, Symmetry Defect, SSIM, PSNR)

## Phase 3: Synthetic Experiment (Section 3.4)

- [x] Implement synthetic data generation with matrices A, B
- [x] Implement MDS + decoder "bridge" (Algorithm 2)
- [x] Implement generator J estimation
- [x] Run Scenario 1: Old Approach
- [x] Run Scenario 2: New Approach (λ=0.5)
- [x] Run Scenario 3: Robustness Test
- [x] Generate scatter plot figures (Editor's requirement)

## Phase 4: MNIST Experiment (Section 3.5)

- [x] Implement CNN model for feature extraction
- [x] Implement rotation augmentation for generator estimation
- [x] Implement memory-efficient solver (avoid explicit Kronecker)
- [x] Run baseline vs equivariant comparison
- [x] Compute SSIM, PSNR metrics
- [x] Generate visualization figures

## Phase 5: Analysis & Documentation

- [x] Run λ sensitivity analysis
- [x] Run ε sensitivity analysis
- [x] Create structured results report
- [x] Create reproducibility checklist
- [x] Self-evaluation and revisions

## Deliverables

- [x] Reproduction methodology document
- [x] Computational experiment plan
- [x] Runnable Python code
- [x] Results report with figures/tables
- [x] Reproducibility checklist

## Results Summary

### MNIST Experiment (200 samples, λ=0.5)
| Metric | Old Approach | New Approach | Improvement |
|--------|--------------|--------------|-------------|
| Robustness MSE | 1.377 | 0.600 | **2.3x better** |
| SSIM | 0.11 | 0.21 | **1.9x better** |
| PSNR | 8.13 dB | 11.44 dB | **+3.3 dB** |

### Epsilon Sensitivity
Optimal ε range: 0.01 - 0.05 radians

