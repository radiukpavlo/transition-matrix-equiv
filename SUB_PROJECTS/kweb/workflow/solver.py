"""
Equivariant Solver for Representation Learning

This module implements the equivariant solver as described in Section 3.2-3.3 of the manuscript.
The solver finds a transition matrix T that minimizes both reconstruction error and symmetry defect.

The objective function is:
    L(T) = ||B^T - T A^T||_F^2 + λ ||T J^A - J^B T||_F^2

This is solved as a linear system without forming the full Kronecker product matrix,
making it scalable to MNIST-size problems (784 × 490).

Author: K-Dense Coding Agent
Date: 2026-01-18
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr, cg
from typing import Optional, Tuple
import time


class EquivariantSolver:
    """
    Solve for equivariant transition matrix T using the combined objective.

    This implements the equivariant solver from Section 3.2-3.3 of the manuscript.
    We solve the linear system:

        [A ⊗ I                           ] vec(T) = [vec(B^T)]
        [√λ (J^A ⊗ I - I ⊗ J^B)         ]          [   0    ]

    using iterative solvers (lsqr or cg) without forming the Kronecker products explicitly.

    Attributes
    ----------
    A : np.ndarray
        Formal model features, shape (N, d_A)
    B : np.ndarray
        Mental model features, shape (N, d_B)
    J_A : np.ndarray
        Generator for formal model, shape (d_A, d_A)
    J_B : np.ndarray
        Generator for mental model, shape (d_B, d_B)
    lambda_reg : float
        Regularization weight for symmetry constraint
    d_A : int
        Dimension of formal model features
    d_B : int
        Dimension of mental model features
    N : int
        Number of samples
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        J_A: np.ndarray,
        J_B: np.ndarray,
        lambda_reg: float = 0.5
    ):
        """
        Initialize the equivariant solver.

        Parameters
        ----------
        A : np.ndarray
            Formal model features, shape (N, d_A)
        B : np.ndarray
            Mental model features, shape (N, d_B)
        J_A : np.ndarray
            Generator for formal model, shape (d_A, d_A)
        J_B : np.ndarray
            Generator for mental model, shape (d_B, d_B)
        lambda_reg : float, default=0.5
            Regularization weight for symmetry constraint
        """
        self.A = A
        self.B = B
        self.J_A = J_A
        self.J_B = J_B
        self.lambda_reg = lambda_reg

        self.N = A.shape[0]
        self.d_A = A.shape[1]
        self.d_B = B.shape[1]

        # Precompute quantities for efficiency
        self.sqrt_lambda = np.sqrt(lambda_reg)

        print(f"Initialized EquivariantSolver:")
        print(f"  Features: A {A.shape}, B {B.shape}")
        print(f"  Generators: J^A {J_A.shape}, J^B {J_B.shape}")
        print(f"  Regularization: λ = {lambda_reg}")
        print(f"  T will have shape ({self.d_B}, {self.d_A})")
        print(f"  Total unknowns: {self.d_B * self.d_A}")

    def _apply_kronecker_A(self, vec_T: np.ndarray) -> np.ndarray:
        """
        Apply (A ⊗ I) to vec(T) without forming the Kronecker product.

        For T of shape (d_B, d_A), we have:
            (A ⊗ I) vec(T) = vec(T A^T)

        Parameters
        ----------
        vec_T : np.ndarray
            Vectorized T, shape (d_B * d_A,)

        Returns
        -------
        result : np.ndarray
            Result of (A ⊗ I) vec(T), shape (N * d_B,)
        """
        T = vec_T.reshape(self.d_B, self.d_A)
        result = T @ self.A.T  # Shape: (d_B, N)
        return result.T.ravel()  # Vectorize to shape (N * d_B,)

    def _apply_symmetry_constraint(self, vec_T: np.ndarray) -> np.ndarray:
        """
        Apply (J^A ⊗ I - I ⊗ J^B) to vec(T) without forming Kronecker products.

        For T of shape (d_B, d_A), we have:
            (J^A ⊗ I) vec(T) = vec(T J^A^T)
            (I ⊗ J^B) vec(T) = vec(J^B T)

        So: (J^A ⊗ I - I ⊗ J^B) vec(T) = vec(T J^A^T - J^B T)

        Parameters
        ----------
        vec_T : np.ndarray
            Vectorized T, shape (d_B * d_A,)

        Returns
        -------
        result : np.ndarray
            Result of (J^A ⊗ I - I ⊗ J^B) vec(T), shape (d_B * d_A,)
        """
        T = vec_T.reshape(self.d_B, self.d_A)
        result = T @ self.J_A.T - self.J_B @ T  # Shape: (d_B, d_A)
        return result.ravel()

    def _matvec(self, vec_T: np.ndarray) -> np.ndarray:
        """
        Matrix-vector product for the combined system.

        This computes:
            [A ⊗ I                           ] vec(T)
            [√λ (J^A ⊗ I - I ⊗ J^B)         ]

        Parameters
        ----------
        vec_T : np.ndarray
            Vectorized T, shape (d_B * d_A,)

        Returns
        -------
        result : np.ndarray
            Concatenated result, shape (N * d_B + d_B * d_A,)
        """
        # First block: (A ⊗ I) vec(T)
        block1 = self._apply_kronecker_A(vec_T)

        # Second block: √λ (J^A ⊗ I - I ⊗ J^B) vec(T)
        block2 = self.sqrt_lambda * self._apply_symmetry_constraint(vec_T)

        return np.concatenate([block1, block2])

    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Adjoint (transpose) matrix-vector product for the combined system.

        This computes:
            [A ⊗ I]^T                          [y1]
            [√λ (J^A ⊗ I - I ⊗ J^B)]^T        [y2]

        where y = [y1; y2], y1 has shape (N * d_B,), y2 has shape (d_B * d_A,)

        Parameters
        ----------
        y : np.ndarray
            Input vector, shape (N * d_B + d_B * d_A,)

        Returns
        -------
        result : np.ndarray
            Result of adjoint operation, shape (d_B * d_A,)
        """
        # Split y into two blocks
        y1 = y[:self.N * self.d_B]  # First block
        y2 = y[self.N * self.d_B:]  # Second block

        # First block: (A ⊗ I)^T y1 = vec(Y1^T A) where Y1 = y1.reshape(N, d_B)
        Y1 = y1.reshape(self.N, self.d_B)
        result1 = (Y1.T @ self.A).ravel()  # Shape: (d_B * d_A,)

        # Second block: √λ (J^A ⊗ I - I ⊗ J^B)^T y2
        # = √λ ((J^A ⊗ I)^T - (I ⊗ J^B)^T) y2
        # = √λ (vec(Y2 J^A) - vec(J^B^T Y2)) where Y2 = y2.reshape(d_B, d_A)
        Y2 = y2.reshape(self.d_B, self.d_A)
        result2 = self.sqrt_lambda * ((Y2 @ self.J_A) - (self.J_B.T @ Y2)).ravel()

        return result1 + result2

    def solve(
        self,
        method: str = 'lsqr',
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve for the equivariant transition matrix T.

        Parameters
        ----------
        method : str, default='lsqr'
            Solver method: 'lsqr' (recommended) or 'cg'
        max_iter : int, default=1000
            Maximum number of iterations
        tol : float, default=1e-6
            Convergence tolerance
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        T : np.ndarray
            Optimal transition matrix, shape (d_B, d_A)
        info : dict
            Solver information (iterations, residual, etc.)
        """
        if verbose:
            print(f"\nSolving equivariant system with {method.upper()}...")
            print(f"  System size: {self.N * self.d_B + self.d_B * self.d_A} × {self.d_B * self.d_A}")
            print(f"  Max iterations: {max_iter}, tolerance: {tol}")

        # Create LinearOperator for the combined system
        shape = (self.N * self.d_B + self.d_B * self.d_A, self.d_B * self.d_A)
        A_op = LinearOperator(shape, matvec=self._matvec, rmatvec=self._rmatvec)

        # Create right-hand side: [vec(B^T); 0]
        rhs = np.concatenate([
            self.B.T.ravel(),
            np.zeros(self.d_B * self.d_A)
        ])

        # Solve the system
        start_time = time.time()

        if method == 'lsqr':
            result = lsqr(
                A_op, rhs,
                iter_lim=max_iter,
                atol=tol,
                btol=tol,
                show=verbose
            )
            vec_T_opt = result[0]
            info = {
                'iterations': result[2],
                'residual_norm': result[3],
                'A_norm': result[4],
                'cond_estimate': result[5],
                'status': result[1]  # 1 = solution found, 2 = least squares solution
            }
        elif method == 'cg':
            # For CG, we solve the normal equations: A^T A x = A^T b
            # This requires a symmetric positive definite system
            def normal_matvec(x):
                return A_op.rmatvec(A_op.matvec(x))

            normal_A = LinearOperator(
                (self.d_B * self.d_A, self.d_B * self.d_A),
                matvec=normal_matvec
            )
            normal_rhs = A_op.rmatvec(rhs)

            vec_T_opt, status = cg(
                normal_A, normal_rhs,
                maxiter=max_iter,
                tol=tol
            )
            info = {
                'status': status,
                'iterations': max_iter if status != 0 else -1
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lsqr' or 'cg'.")

        elapsed = time.time() - start_time

        # Reshape solution
        T_opt = vec_T_opt.reshape(self.d_B, self.d_A)

        if verbose:
            print(f"  Solver completed in {elapsed:.2f} seconds")
            if method == 'lsqr':
                print(f"  Iterations: {info['iterations']}")
                print(f"  Residual norm: {info['residual_norm']:.6e}")
                print(f"  Status: {info['status']}")

        # Compute objectives
        info['elapsed_time'] = elapsed
        info['reconstruction_error'] = np.linalg.norm(self.B.T - T_opt @ self.A.T, 'fro') ** 2
        info['symmetry_defect'] = np.linalg.norm(
            T_opt @ self.J_A - self.J_B @ T_opt, 'fro'
        ) ** 2
        info['total_objective'] = (
            info['reconstruction_error'] + self.lambda_reg * info['symmetry_defect']
        )

        if verbose:
            print(f"\nObjective breakdown:")
            print(f"  Reconstruction error: {info['reconstruction_error']:.6e}")
            print(f"  Symmetry defect: {info['symmetry_defect']:.6e}")
            print(f"  Total objective: {info['total_objective']:.6e}")

        return T_opt, info


def compute_objective(
    T: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
    lambda_reg: float
) -> Tuple[float, float, float]:
    """
    Compute the equivariant objective function for a given T.

    L(T) = ||B^T - T A^T||_F^2 + λ ||T J^A - J^B T||_F^2

    Parameters
    ----------
    T : np.ndarray
        Transition matrix, shape (d_B, d_A)
    A : np.ndarray
        Formal model features, shape (N, d_A)
    B : np.ndarray
        Mental model features, shape (N, d_B)
    J_A : np.ndarray
        Generator for formal model, shape (d_A, d_A)
    J_B : np.ndarray
        Generator for mental model, shape (d_B, d_B)
    lambda_reg : float
        Regularization weight

    Returns
    -------
    total : float
        Total objective value
    recon_error : float
        Reconstruction error component
    sym_defect : float
        Symmetry defect component
    """
    recon_error = np.linalg.norm(B.T - T @ A.T, 'fro') ** 2
    sym_defect = np.linalg.norm(T @ J_A - J_B @ T, 'fro') ** 2
    total = recon_error + lambda_reg * sym_defect

    return total, recon_error, sym_defect


if __name__ == "__main__":
    # Quick test with small synthetic data
    print("=" * 60)
    print("Testing equivariant solver module")
    print("=" * 60)

    # Create small test problem
    np.random.seed(42)
    N, d_A, d_B = 20, 5, 8

    A = np.random.randn(N, d_A)
    B = np.random.randn(N, d_B)

    # Create antisymmetric generators (representing rotations)
    J_A = np.random.randn(d_A, d_A)
    J_A = (J_A - J_A.T) / 2  # Make antisymmetric

    J_B = np.random.randn(d_B, d_B)
    J_B = (J_B - J_B.T) / 2  # Make antisymmetric

    print(f"\nTest problem: A {A.shape}, B {B.shape}")
    print(f"Generators: J^A {J_A.shape}, J^B {J_B.shape}")

    # Solve with equivariant solver
    solver = EquivariantSolver(A, B, J_A, J_B, lambda_reg=0.5)
    T_new, info = solver.solve(method='lsqr', max_iter=500)

    print(f"\nSolution T_new shape: {T_new.shape}")

    # Compare with baseline (pseudoinverse)
    T_old = np.linalg.pinv(A.T) @ B.T
    T_old = T_old.T

    obj_old, recon_old, sym_old = compute_objective(T_old, A, B, J_A, J_B, 0.5)
    obj_new, recon_new, sym_new = compute_objective(T_new, A, B, J_A, J_B, 0.5)

    print(f"\nComparison:")
    print(f"  T_old: objective={obj_old:.6e}, recon={recon_old:.6e}, sym={sym_old:.6e}")
    print(f"  T_new: objective={obj_new:.6e}, recon={recon_new:.6e}, sym={sym_new:.6e}")
    print(f"  Improvement: {(obj_old - obj_new)/obj_old * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Equivariant solver module test complete!")
    print("=" * 60)
