"""
Lie algebra generator estimation.

Implements generator estimation as described in Section 3.1 and Algorithm 2
of the manuscript. For a Lie group G acting on data, we estimate the
infinitesimal generators J^A and J^B in the feature spaces.

For SO(2) rotations, the generator describes how features change under
infinitesimal rotations: Δa ≈ J^A @ a for small angle ε.
"""

import numpy as np
from numpy.linalg import pinv, lstsq
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional


def rotate_2d_points(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate 2D points by a given angle using SO(2) rotation matrix.
    
    R(θ) = [[cos(θ), -sin(θ)],
            [sin(θ),  cos(θ)]]
    
    Parameters
    ----------
    points : np.ndarray
        Points of shape (m, 2)
    angle : float
        Rotation angle in radians
    
    Returns
    -------
    np.ndarray
        Rotated points of shape (m, 2)
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    return points @ R.T


def estimate_generator(A: np.ndarray, A_rot: np.ndarray, 
                       epsilon: float) -> np.ndarray:
    """
    Estimate Lie algebra generator from original and rotated features.
    
    Given features A and rotated features A_rot where:
        A_rot ≈ A @ (I + ε J^T)
    
    We solve for J via:
        ΔA = (A_rot - A) / ε ≈ A @ J^T
        J^T = pinv(A) @ ΔA
    
    Parameters
    ----------
    A : np.ndarray
        Original feature matrix of shape (m, k)
    A_rot : np.ndarray
        Rotated feature matrix of shape (m, k)
    epsilon : float
        Small rotation angle used (in radians)
    
    Returns
    -------
    np.ndarray
        Estimated generator J of shape (k, k)
    
    Notes
    -----
    Dimension check:
    - A, A_rot: (m, k)
    - ΔA: (m, k)
    - J^T = pinv(A) @ ΔA: (k, m) @ (m, k) = (k, k)
    - J: (k, k)
    """
    # Compute numerical derivative
    delta_A = (A_rot - A) / epsilon
    
    # Solve A @ J^T = ΔA for J^T
    J_T = pinv(A) @ delta_A
    
    # Transpose to get J
    J = J_T.T
    
    return J


def estimate_generator_lstsq(A: np.ndarray, A_rot: np.ndarray,
                             epsilon: float) -> Tuple[np.ndarray, dict]:
    """
    Estimate generator using least squares with condition reporting.
    
    Parameters
    ----------
    A : np.ndarray
        Original features of shape (m, k)
    A_rot : np.ndarray
        Rotated features of shape (m, k)
    epsilon : float
        Rotation angle in radians
    
    Returns
    -------
    J : np.ndarray
        Generator matrix of shape (k, k)
    info : dict
        Diagnostic info including residual and condition
    """
    delta_A = (A_rot - A) / epsilon
    
    # Use lstsq for better conditioning info
    J_T, residuals, rank, s = lstsq(A, delta_A, rcond=None)
    J = J_T.T
    
    info = {
        'residual_norm': np.sqrt(residuals.sum()) if len(residuals) > 0 else 0.0,
        'matrix_rank': rank,
        'singular_values': s,
        'condition_number': s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf,
    }
    
    return J, info


class SyntheticBridge:
    """
    Algorithm 2: MDS + decoder bridge for synthetic data.
    
    When we have abstract feature matrices A, B without input images,
    we use 2D visualization space as a "control bridge" to define
    rotations and estimate generators.
    
    Steps:
    1. Reduce A → A_2d using MDS
    2. Train decoder: 2D → k-dim
    3. Rotate in 2D, map back to k-dim
    4. Estimate J from original and rotated features
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the synthetic bridge.
        
        Parameters
        ----------
        random_state : int
            Random seed for MDS reproducibility
        """
        self.random_state = random_state
        self.mds = None
        self.decoder = None
        self.A_2d = None
    
    def fit(self, A: np.ndarray) -> 'SyntheticBridge':
        """
        Fit MDS reduction and decoder.
        
        Parameters
        ----------
        A : np.ndarray
            Feature matrix of shape (m, k)
        
        Returns
        -------
        self
        """
        # Step 1: MDS reduction to 2D
        self.mds = MDS(n_components=2, random_state=self.random_state,
                       normalized_stress='auto')
        self.A_2d = self.mds.fit_transform(A)
        
        # Step 2: Train inverse mapping (decoder)
        self.decoder = LinearRegression()
        self.decoder.fit(self.A_2d, A)
        
        return self
    
    def rotate_and_reconstruct(self, angle: float) -> np.ndarray:
        """
        Rotate 2D points and reconstruct to original dimension.
        
        Parameters
        ----------
        angle : float
            Rotation angle in radians
        
        Returns
        -------
        np.ndarray
            Rotated features in original dimension (m, k)
        """
        if self.A_2d is None:
            raise RuntimeError("Must call fit() first")
        
        # Step 3: Rotate in 2D
        A_2d_rot = rotate_2d_points(self.A_2d, angle)
        
        # Step 4: Map back via decoder
        A_rot = self.decoder.predict(A_2d_rot)
        
        return A_rot
    
    def estimate_generator(self, A: np.ndarray, 
                          epsilon: float = 0.01) -> np.ndarray:
        """
        Estimate generator J for feature matrix A.
        
        Complete Algorithm 2 implementation.
        
        Parameters
        ----------
        A : np.ndarray
            Feature matrix of shape (m, k)
        epsilon : float
            Small rotation angle in radians
        
        Returns
        -------
        np.ndarray
            Generator J of shape (k, k)
        """
        # Fit bridge if not already done
        if self.A_2d is None:
            self.fit(A)
        
        # Get rotated features
        A_rot = self.rotate_and_reconstruct(epsilon)
        
        # Estimate generator
        J = estimate_generator(A, A_rot, epsilon)
        
        return J
    
    def get_2d_projection(self) -> np.ndarray:
        """Get the 2D MDS projection."""
        return self.A_2d


def rotate_image(image: np.ndarray, angle: float,
                 interpolation: str = 'bilinear') -> np.ndarray:
    """
    Rotate a 2D image by given angle.
    
    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, C)
    angle : float
        Rotation angle in radians
    interpolation : str
        Interpolation method ('nearest', 'bilinear')
    
    Returns
    -------
    np.ndarray
        Rotated image of same shape
    """
    from scipy.ndimage import rotate as scipy_rotate
    
    # Convert radians to degrees for scipy
    angle_deg = np.degrees(angle)
    
    # Rotate with appropriate settings
    rotated = scipy_rotate(image, angle_deg, reshape=False, 
                          order=1 if interpolation == 'bilinear' else 0,
                          mode='constant', cval=0.0)
    
    return rotated


def estimate_generator_from_images(images: np.ndarray,
                                   feature_fn: callable,
                                   epsilon: float = 0.01) -> np.ndarray:
    """
    Estimate generator from actual images using rotation augmentation.
    
    For MNIST: apply small rotation to images, extract features,
    and compute generator from feature changes.
    
    Parameters
    ----------
    images : np.ndarray
        Batch of images, shape (m, H, W) or (m, H, W, C)
    feature_fn : callable
        Function that extracts features: images → features (m, k)
    epsilon : float
        Small rotation angle in radians
    
    Returns
    -------
    np.ndarray
        Generator J of shape (k, k)
    """
    # Get original features
    A = feature_fn(images)
    
    # Rotate all images by small angle
    images_rot = np.array([rotate_image(img, epsilon) for img in images])
    
    # Get rotated features
    A_rot = feature_fn(images_rot)
    
    # Estimate generator
    J = estimate_generator(A, A_rot, epsilon)
    
    return J
