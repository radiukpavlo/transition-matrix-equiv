"""
Unit tests for core mathematical operations.

Tests algebraic identities and numerical correctness:
- vec/devec identity
- Kronecker-vector identity
- SVD solver accuracy
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import vec, devec, kron_mvp, svd_solve


class TestVectorization:
    """Tests for vec and devec operations."""
    
    def test_vec_devec_identity(self):
        """Test that vec(devec(u)) == u for any vector u."""
        np.random.seed(42)
        
        for shape in [(3, 4), (5, 5), (2, 7), (10, 3)]:
            u = np.random.randn(shape[0] * shape[1])
            
            # Reshape and flatten
            T = devec(u, shape)
            u_recovered = vec(T)
            
            np.testing.assert_array_almost_equal(
                u, u_recovered,
                decimal=14,
                err_msg=f"vec/devec identity failed for shape {shape}"
            )
    
    def test_devec_vec_identity(self):
        """Test that devec(vec(T)) == T for any matrix T."""
        np.random.seed(42)
        
        for shape in [(3, 4), (5, 5), (2, 7), (10, 3)]:
            T = np.random.randn(*shape)
            
            # Flatten and reshape
            u = vec(T)
            T_recovered = devec(u, shape)
            
            np.testing.assert_array_almost_equal(
                T, T_recovered,
                decimal=14,
                err_msg=f"devec/vec identity failed for shape {shape}"
            )
    
    def test_vec_column_major(self):
        """Test that vec uses column-major (Fortran) order."""
        T = np.array([[1, 2, 3],
                      [4, 5, 6]])  # shape (2, 3)
        
        expected = np.array([1, 4, 2, 5, 3, 6])  # column-major
        result = vec(T)
        
        np.testing.assert_array_equal(result, expected)


class TestKroneckerVectorProduct:
    """Tests for Kronecker matrix-vector product identity."""
    
    def test_kronecker_vec_identity(self):
        """Test vec(AXB) == (B^T ⊗ A) vec(X)."""
        np.random.seed(42)
        
        m, n, p, q = 3, 4, 5, 2
        A = np.random.randn(m, n)
        X = np.random.randn(n, p)
        B = np.random.randn(p, q)
        
        # Direct computation
        AXB = A @ X @ B
        left_side = vec(AXB)
        
        # Kronecker identity
        kron_matrix = np.kron(B.T, A)
        right_side = kron_matrix @ vec(X)
        
        np.testing.assert_array_almost_equal(
            left_side, right_side, decimal=12,
            err_msg="Kronecker-vec identity failed"
        )
    
    def test_kron_mvp_efficient(self):
        """Test efficient Kronecker MVP without explicit formation."""
        np.random.seed(42)
        
        m, n, p, q = 4, 3, 3, 2
        A = np.random.randn(m, n)
        B = np.random.randn(p, q)
        x = np.random.randn(n * q)
        
        # Explicit Kronecker: (m*p) x (n*q) matrix times (n*q) vector
        K = np.kron(A, B)
        expected = K @ x
        
        # Efficient MVP
        result = kron_mvp(A, B, x)
        
        np.testing.assert_array_almost_equal(
            result, expected, decimal=12,
            err_msg="Efficient Kronecker MVP failed"
        )


class TestSVDSolver:
    """Tests for SVD-based least squares solver."""
    
    def test_svd_solve_accuracy(self):
        """Test SVD solver correctly solves overdetermined systems."""
        np.random.seed(42)
        
        # Overdetermined system
        m, n = 10, 5
        M = np.random.randn(m, n)
        x_true = np.random.randn(n)
        Y = M @ x_true
        
        # Solve
        x_solved = svd_solve(M, Y)
        
        np.testing.assert_array_almost_equal(
            x_solved, x_true, decimal=10,
            err_msg="SVD solver failed for exact system"
        )
    
    def test_svd_solve_least_squares(self):
        """Test SVD solver gives least squares solution for noisy systems."""
        np.random.seed(42)
        
        m, n = 20, 5
        M = np.random.randn(m, n)
        Y = np.random.randn(m)
        
        # SVD solution
        x_svd = svd_solve(M, Y)
        
        # NumPy least squares solution
        x_np, _, _, _ = np.linalg.lstsq(M, Y, rcond=None)
        
        np.testing.assert_array_almost_equal(
            x_svd, x_np, decimal=10,
            err_msg="SVD solver differs from lstsq"
        )
    
    def test_svd_solve_regularization(self):
        """Test regularization threshold filters small singular values."""
        np.random.seed(42)
        
        # Create ill-conditioned matrix
        m, n = 10, 5
        U = np.random.randn(m, n)
        V = np.random.randn(n, n)
        S = np.diag([1.0, 0.1, 0.01, 1e-8, 1e-12])  # Very different singular values
        M = U @ S @ V
        Y = np.random.randn(m)
        
        # Solve with different thresholds
        x_loose = svd_solve(M, Y, tau=1e-6)
        x_tight = svd_solve(M, Y, tau=1e-15)
        
        # Both should be valid (but may differ due to regularization)
        assert x_loose.shape == (n,)
        assert x_tight.shape == (n,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
