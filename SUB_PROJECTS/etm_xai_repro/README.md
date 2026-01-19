# Equivariant Transition Matrix (ETM) Reproducibility Project

This repository is a **plug-and-play** reproduction package for the manuscript's
ETM method and its baseline comparator.

It contains:

- A fully offline **synthetic** experiment (Section 3.4)
- A fully specified **MNIST** experiment (Section 3.5) that downloads MNIST in
  IDX format (no `torchvision` required) and reproduces the baseline PDF setup
  (subsampling sizes, CNN architecture, SSIM/PSNR evaluation)
- A **10‑section Reproducibility Appendix** that can be appended verbatim to the
  paper (`REPRODUCIBILITY_APPENDIX.md`)

## Quick start (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run everything (synthetic runs offline; MNIST runs only if data are available)
python scripts/run_all.py
```

Artifacts are written under `outputs/`.

## MNIST data

The MNIST script expects the four IDX gzip files under:

```bash
data/mnist/
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
```

In a normal environment with internet access, you can fetch them via:

```bash
python scripts/run_mnist.py --download
```

If you are running in a restricted environment (no outbound net), download those
files once elsewhere and place them into `data/mnist/`.

## Reproducing the paper figures

- **Synthetic:**
  - `outputs/synthetic/synthetic_metrics.json`
  - `outputs/synthetic/synthetic_scatter_rotated_old_vs_new.png`

- **MNIST (after data download):**
  - `outputs/mnist/metrics_baseline_unrotated.csv`
  - `outputs/mnist/metrics_equivariant_unrotated.csv`
  - `outputs/mnist/metrics_baseline_rotated.csv`
  - `outputs/mnist/metrics_equivariant_rotated.csv`
  - `outputs/mnist/mnist_baseline_recon_examples.png`
  - `outputs/mnist/mnist_equivariant_recon_examples.png`
  - `outputs/mnist/mnist_tradeoff_lambda.png`

## Notes about this sandbox run

The provided ZIP includes the full project. In the execution sandbox used to
generate this artifact, **outbound network from Python is disabled**, so MNIST
cannot be downloaded automatically during the run. The script therefore skips
MNIST unless data are already present.

The synthetic experiment is executed and its outputs are included.
