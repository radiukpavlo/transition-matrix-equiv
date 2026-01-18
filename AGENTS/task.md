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
- [ ] Run baseline vs equivariant comparison
- [ ] Compute SSIM, PSNR metrics
- [ ] Generate visualization figures

## Phase 5: Analysis & Documentation

- [x] Run λ sensitivity analysis
- [ ] Run ε sensitivity analysis
- [x] Create structured results report
- [x] Create reproducibility checklist
- [ ] Self-evaluation and revisions

## Deliverables

- [x] Reproduction methodology document
- [x] Computational experiment plan
- [x] Runnable Python code
- [x] Results report with figures/tables
- [x] Reproducibility checklist
