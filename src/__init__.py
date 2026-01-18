# Equivariant Transition Matrices
# Reproducing: "Equivariant Transition Matrices for Explainable Deep Learning"

from .core import vec, devec, kron_mvp, svd_solve
from .baseline import compute_baseline_T
from .generators import estimate_generator, rotate_2d_points
from .equivariant import compute_equivariant_T, compute_equivariant_T_iterative
from .metrics import fidelity_mse, symmetry_defect, compute_ssim, compute_psnr

__version__ = "1.0.0"
__all__ = [
    "vec", "devec", "kron_mvp", "svd_solve",
    "compute_baseline_T",
    "estimate_generator", "rotate_2d_points",
    "compute_equivariant_T", "compute_equivariant_T_iterative",
    "fidelity_mse", "symmetry_defect", "compute_ssim", "compute_psnr",
]
