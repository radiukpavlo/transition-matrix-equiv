# K-Dense Web Research Reproduction Task

## Phase 1: Planning & Setup

- [x] Read and understand PROMPT.md requirements
- [x] Create implementation plan
- [x] Request user approval for plan

## Phase 2: Core Implementation

- [/] Create project structure (src/, tests/, figures/, results/)
- [ ] Implement core utilities (vectorization, Kronecker products, SVD solver)
- [ ] Implement baseline transition matrix (Old Approach)
- [ ] Implement generator estimation (Lie algebra)
- [ ] Implement equivariant transition matrix (New Approach)
- [ ] Implement evaluation metrics (MSE, Symmetry Defect, SSIM, PSNR)

## Phase 3: Synthetic Experiment (Section 3.4)

- [ ] Implement synthetic data generation with matrices A, B
- [ ] Implement MDS + decoder "bridge" (Algorithm 2)
- [ ] Implement generator J estimation
- [ ] Run Scenario 1: Old Approach
- [ ] Run Scenario 2: New Approach (λ=0.5)
- [ ] Run Scenario 3: Robustness Test
- [ ] Generate scatter plot figures (Editor's requirement)

## Phase 4: MNIST Experiment (Section 3.5)

- [ ] Implement CNN model for feature extraction
- [ ] Implement rotation augmentation for generator estimation
- [ ] Implement memory-efficient solver (avoid explicit Kronecker)
- [ ] Run baseline vs equivariant comparison
- [ ] Compute SSIM, PSNR metrics
- [ ] Generate visualization figures

## Phase 5: Analysis & Documentation

- [ ] Run λ sensitivity analysis
- [ ] Run ε sensitivity analysis
- [ ] Create structured results report
- [ ] Create reproducibility checklist
- [ ] Self-evaluation and revisions

## Deliverables

- [ ] Reproduction methodology document
- [ ] Computational experiment plan
- [ ] Runnable Python code
- [ ] Results report with figures/tables
- [ ] Reproducibility checklist
