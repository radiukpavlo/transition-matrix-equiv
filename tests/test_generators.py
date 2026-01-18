"""
Unit tests for Lie algebra generator estimation.

Tests:
- SO(2) generator antisymmetry
- Small epsilon stability
- Generator estimation correctness
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators import (
    rotate_2d_points,
    estimate_generator,
    estimate_generator_lstsq,
    SyntheticBridge
)


class TestRotation:
    """Tests for SO(2) rotation operations."""
    
    def test_rotate_identity(self):
        """Test that zero rotation is identity."""
        points = np.array([[1, 0], [0, 1], [1, 1]])
        rotated = rotate_2d_points(points, 0)
        np.testing.assert_array_almost_equal(points, rotated, decimal=14)
    
    def test_rotate_90_degrees(self):
        """Test 90-degree rotation."""
        points = np.array([[1, 0], [0, 1]])
        rotated = rotate_2d_points(points, np.pi / 2)
        
        expected = np.array([[0, 1], [-1, 0]])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=14)
    
    def test_rotate_composition(self):
        """Test that two rotations compose correctly."""
        np.random.seed(42)
        points = np.random.randn(10, 2)
        
        theta1, theta2 = 0.3, 0.5
        
        # Compose rotations
        r1 = rotate_2d_points(points, theta1)
        r12 = rotate_2d_points(r1, theta2)
        
        # Single rotation
        r_direct = rotate_2d_points(points, theta1 + theta2)
        
        np.testing.assert_array_almost_equal(r12, r_direct, decimal=14)


class TestGeneratorEstimation:
    """Tests for Lie algebra generator estimation."""
    
    def test_so2_generator_structure(self):
        """Test that SO(2) generator is antisymmetric in 2D."""
        np.random.seed(42)
        
        # Simple 2D rotation case
        m = 20
        points_2d = np.random.randn(m, 2)
        
        epsilon = 0.001
        points_rot = rotate_2d_points(points_2d, epsilon)
        
        J = estimate_generator(points_2d, points_rot, epsilon)
        
        # SO(2) generator should be antisymmetric: J + J^T = 0
        antisymmetric_part = J + J.T
        np.testing.assert_array_almost_equal(
            antisymmetric_part, np.zeros((2, 2)), decimal=2,
            err_msg="SO(2) generator should be antisymmetric"
        )
    
    def test_so2_generator_eigenvalues(self):
        """Test that SO(2) generator has purely imaginary eigenvalues."""
        np.random.seed(42)
        
        m = 50
        points_2d = np.random.randn(m, 2)
        epsilon = 0.001
        points_rot = rotate_2d_points(points_2d, epsilon)
        
        J = estimate_generator(points_2d, points_rot, epsilon)
        
        eigenvalues = np.linalg.eigvals(J)
        
        # Real parts should be close to zero
        real_parts = np.real(eigenvalues)
        np.testing.assert_array_almost_equal(
            real_parts, np.zeros(2), decimal=2,
            err_msg="SO(2) generator eigenvalues should be purely imaginary"
        )
    
    def test_small_epsilon_stability(self):
        """Test generator estimation is stable for small epsilon."""
        np.random.seed(42)
        
        m = 30
        points_2d = np.random.randn(m, 2)
        
        epsilons = [0.1, 0.01, 0.001]
        generators = []
        
        for eps in epsilons:
            points_rot = rotate_2d_points(points_2d, eps)
            J = estimate_generator(points_2d, points_rot, eps)
            generators.append(J)
        
        # Generators should be similar across different epsilon values
        for i in range(len(generators) - 1):
            diff = np.linalg.norm(generators[i] - generators[i+1], 'fro')
            assert diff < 0.5, f"Generators differ too much: {diff}"


class TestSyntheticBridge:
    """Tests for Algorithm 2: MDS + decoder bridge."""
    
    def test_bridge_fit(self):
        """Test that bridge fits without error."""
        np.random.seed(42)
        A = np.random.randn(15, 5)
        
        bridge = SyntheticBridge(random_state=42)
        bridge.fit(A)
        
        assert bridge.A_2d is not None
        assert bridge.A_2d.shape == (15, 2)
        assert bridge.decoder is not None
    
    def test_bridge_rotate_and_reconstruct(self):
        """Test rotation and reconstruction through bridge."""
        np.random.seed(42)
        A = np.random.randn(15, 5)
        
        bridge = SyntheticBridge(random_state=42)
        bridge.fit(A)
        
        # Zero rotation should give similar (not identical due to MDS/regression loss)
        # MDS reduces to 2D and linear regression reconstructs, so significant loss expected
        A_rot_0 = bridge.rotate_and_reconstruct(0)
        diff = np.linalg.norm(A - A_rot_0, 'fro') / np.linalg.norm(A, 'fro')
        assert diff < 0.8, f"Zero rotation differs too much: {diff}"
    
    def test_bridge_generator_estimation(self):
        """Test generator estimation through bridge."""
        np.random.seed(42)
        A = np.random.randn(15, 5)
        
        bridge = SyntheticBridge(random_state=42)
        J = bridge.estimate_generator(A, epsilon=0.01)
        
        # Generator should have correct shape
        assert J.shape == (5, 5)
        
        # Should not be all zeros
        assert np.linalg.norm(J, 'fro') > 0.01


class TestGeneratorLstSq:
    """Tests for least squares generator estimation with diagnostics."""
    
    def test_lstsq_same_as_pinv(self):
        """Test that lstsq method gives same generator as pinv."""
        np.random.seed(42)
        
        m = 30
        points = np.random.randn(m, 3)
        epsilon = 0.01
        
        # Create rotation in higher dimension (simplified)
        rotation = np.eye(3) + epsilon * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        points_rot = points @ rotation.T
        
        J_pinv = estimate_generator(points, points_rot, epsilon)
        J_lstsq, info = estimate_generator_lstsq(points, points_rot, epsilon)
        
        np.testing.assert_array_almost_equal(
            J_pinv, J_lstsq, decimal=10,
            err_msg="LSTSQ and PINV methods should give same result"
        )
    
    def test_lstsq_returns_info(self):
        """Test that lstsq method returns diagnostic info."""
        np.random.seed(42)
        
        points = np.random.randn(20, 3)
        epsilon = 0.01
        points_rot = points  # identity transformation
        
        J, info = estimate_generator_lstsq(points, points_rot, epsilon)
        
        assert 'matrix_rank' in info
        assert 'singular_values' in info
        assert 'condition_number' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
