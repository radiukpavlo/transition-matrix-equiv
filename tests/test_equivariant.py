"""
Unit tests for equivariant transition matrix solver.

Tests:
- λ=0 gives baseline solution
- Higher λ → lower symmetry defect
- Iterative solver equivalence to explicit solver
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline import compute_baseline_T, predict_mm_features
from src.equivariant import (
    compute_equivariant_T,
    compute_equivariant_T_iterative,
    build_fidelity_system,
    build_symmetry_system
)
from src.metrics import fidelity_mse, symmetry_defect


class TestFidelitySystem:
    """Tests for fidelity system construction."""
    
    def test_fidelity_dimensions(self):
        """Test fidelity system has correct dimensions."""
        m, k, l = 10, 5, 4
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        
        M_fid, Y_fid = build_fidelity_system(A, B)
        
        assert M_fid.shape == (m * l, k * l)
        assert Y_fid.shape == (m * l,)
    
    def test_fidelity_kronecker_structure(self):
        """Test fidelity matrix has correct Kronecker structure."""
        m, k, l = 5, 3, 2
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        
        M_fid, Y_fid = build_fidelity_system(A, B)
        
        # Verify it equals A ⊗ I_l
        I_l = np.eye(l)
        expected = np.kron(A, I_l)
        
        np.testing.assert_array_almost_equal(M_fid, expected)


class TestSymmetrySystem:
    """Tests for symmetry constraint system construction."""
    
    def test_symmetry_dimensions(self):
        """Test symmetry constraint matrix has correct dimensions."""
        k, l = 5, 4
        J_A = np.random.randn(k, k)
        J_B = np.random.randn(l, l)
        
        K = build_symmetry_system(J_A, J_B, k, l)
        
        assert K.shape == (k * l, k * l)
    
    def test_symmetry_structure(self):
        """Test symmetry matrix has correct Kronecker structure."""
        k, l = 3, 2
        J_A = np.random.randn(k, k)
        J_B = np.random.randn(l, l)
        
        K = build_symmetry_system(J_A, J_B, k, l)
        
        # Verify it equals (J_A^T ⊗ I_l) - (I_k ⊗ J_B)
        I_l = np.eye(l)
        I_k = np.eye(k)
        expected = np.kron(J_A.T, I_l) - np.kron(I_k, J_B)
        
        np.testing.assert_array_almost_equal(K, expected)


class TestEquivariantSolver:
    """Tests for equivariant transition matrix computation."""
    
    def test_lambda_zero_equals_baseline(self):
        """Test that λ=0 gives the baseline (fidelity-only) solution."""
        np.random.seed(42)
        
        m, k, l = 15, 5, 4
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        J_A = np.random.randn(k, k) * 0.1
        J_B = np.random.randn(l, l) * 0.1
        
        T_baseline = compute_baseline_T(A, B)
        T_equivariant = compute_equivariant_T(A, B, J_A, J_B, lambda_=0)
        
        np.testing.assert_array_almost_equal(
            T_baseline, T_equivariant, decimal=6,
            err_msg="λ=0 should give baseline solution"
        )
    
    def test_symmetry_defect_decreases_with_lambda(self):
        """Test that higher λ gives lower symmetry defect."""
        np.random.seed(42)
        
        m, k, l = 15, 5, 4
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        
        # Create antisymmetric generators (like SO(2))
        J_A = np.random.randn(k, k)
        J_A = (J_A - J_A.T) * 0.5  # Make antisymmetric
        J_B = np.random.randn(l, l)
        J_B = (J_B - J_B.T) * 0.5  # Make antisymmetric
        
        lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
        sym_defects = []
        
        for lam in lambdas:
            T = compute_equivariant_T(A, B, J_A, J_B, lambda_=lam)
            sym = symmetry_defect(T, J_A, J_B)
            sym_defects.append(sym)
        
        # Check monotonic decrease (approximately)
        for i in range(len(sym_defects) - 1):
            # Allow some tolerance due to numerical issues
            assert sym_defects[i+1] <= sym_defects[i] * 1.1, \
                f"Symmetry defect should decrease with λ: {sym_defects}"
    
    def test_fidelity_increases_with_lambda(self):
        """Test that higher λ gives higher fidelity error."""
        np.random.seed(42)
        
        m, k, l = 15, 5, 4
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        J_A = np.random.randn(k, k) * 0.5
        J_B = np.random.randn(l, l) * 0.5
        
        lambdas = [0.001, 0.1, 1.0, 10.0]
        mse_values = []
        
        for lam in lambdas:
            T = compute_equivariant_T(A, B, J_A, J_B, lambda_=lam)
            B_pred = predict_mm_features(A, T)
            mse = fidelity_mse(B, B_pred)
            mse_values.append(mse)
        
        # Check monotonic increase (approximately)
        for i in range(len(mse_values) - 1):
            assert mse_values[i+1] >= mse_values[i] * 0.9, \
                f"MSE should increase with λ: {mse_values}"


class TestIterativeSolver:
    """Tests for iterative (LSQR) solver."""
    
    def test_iterative_matches_explicit(self):
        """Test iterative solver gives same result as explicit for small problems."""
        np.random.seed(42)
        
        m, k, l = 15, 5, 4
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        J_A = np.random.randn(k, k) * 0.1
        J_B = np.random.randn(l, l) * 0.1
        
        T_explicit = compute_equivariant_T(A, B, J_A, J_B, lambda_=0.5)
        T_iterative, info = compute_equivariant_T_iterative(
            A, B, J_A, J_B, lambda_=0.5, atol=1e-10, btol=1e-10
        )
        
        # Allow slightly relaxed tolerance since iterative solvers have different convergence
        np.testing.assert_array_almost_equal(
            T_explicit, T_iterative, decimal=2,
            err_msg="Iterative and explicit solvers should give similar results"
        )
    
    def test_iterative_returns_info(self):
        """Test iterative solver returns useful diagnostics."""
        np.random.seed(42)
        
        m, k, l = 15, 5, 4
        A = np.random.randn(m, k)
        B = np.random.randn(m, l)
        J_A = np.random.randn(k, k) * 0.1
        J_B = np.random.randn(l, l) * 0.1
        
        T, info = compute_equivariant_T_iterative(A, B, J_A, J_B, lambda_=0.5)
        
        assert 'iterations' in info
        assert 'residual_norm' in info
        assert info['iterations'] > 0


class TestWithSyntheticData:
    """Tests using the actual synthetic data from the manuscript."""
    
    def test_on_manuscript_data(self):
        """Test on the actual A, B matrices from Appendix 1.1."""
        from experiments.synthetic.data import get_matrix_A, get_matrix_B
        from src.generators import SyntheticBridge
        
        A = get_matrix_A()
        B = get_matrix_B()
        
        # Estimate generators
        bridge_A = SyntheticBridge(random_state=42)
        J_A = bridge_A.estimate_generator(A, epsilon=0.01)
        
        bridge_B = SyntheticBridge(random_state=42)
        J_B = bridge_B.estimate_generator(B, epsilon=0.01)
        
        # Compute both T_old and T_new
        T_old = compute_baseline_T(A, B)
        T_new = compute_equivariant_T(A, B, J_A, J_B, lambda_=0.5)
        
        # Both should have correct shapes
        assert T_old.shape == (4, 5)
        assert T_new.shape == (4, 5)
        
        # T_new should have lower symmetry defect than T_old
        sym_old = symmetry_defect(T_old, J_A, J_B)
        sym_new = symmetry_defect(T_new, J_A, J_B)
        
        assert sym_new < sym_old, \
            f"New approach should have lower sym defect: {sym_new} vs {sym_old}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
