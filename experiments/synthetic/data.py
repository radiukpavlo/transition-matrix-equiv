"""
Synthetic data for Section 3.4 experiments.

Data from manuscript Appendix 1.1:
- Matrix A ∈ ℝ^(15×5): FM features (5 dimensions)
- Matrix B ∈ ℝ^(15×4): MM features (4 dimensions)
- 15 samples divided into 3 classes (5 samples each)
"""

import numpy as np


def get_matrix_A() -> np.ndarray:
    """
    Get synthetic FM feature matrix A from Appendix 1.1.
    
    Returns
    -------
    np.ndarray
        Matrix A of shape (15, 5)
    """
    A = np.array([
        [2.8, -1.8, -2.8, 1.3, 0.4],
        [2.9, -1.9, -2.9, 1.4, 0.5],
        [3.0, -2.0, -3.0, 1.5, 0.6],
        [3.1, -2.1, -3.1, 1.6, 0.7],
        [3.2, -2.2, -3.2, 1.7, 0.8],
        [-1.6, -2.5, 1.5, 0.2, 0.6],
        [-1.3, -2.7, 1.3, 0.4, 0.8],
        [-1.0, -3.0, 1.5, 0.6, 1.0],
        [-0.7, -3.2, 1.7, 0.8, 1.2],
        [-0.5, -3.5, 1.9, 1.0, 1.4],
        [1.2, -1.2, 0.7, -0.3, -2.8],
        [1.1, -1.1, 0.8, -0.4, -2.9],
        [1.0, -1.0, 0.84, -0.44, -3.0],  # Note: 0.8(4) = 0.84, -0.(4) = -0.44
        [0.9, -0.9, 0.85, -0.45, -3.1],
        [0.8, -0.8, 0.9, -0.5, -3.2],
    ])
    return A


def get_matrix_B() -> np.ndarray:
    """
    Get synthetic MM feature matrix B from Appendix 1.1.
    
    Returns
    -------
    np.ndarray
        Matrix B of shape (15, 4)
    """
    B = np.array([
        [-1.979394104, 1.959307524, -1.381119943, -1.72964],
        [-1.974921385, 1.94850558, -1.726609792, -1.76121],
        [-1.843907868, 1.99818664, -1.912855282, -1.97511],
        [-1.998625355, 1.999671808, -1.998443276, -1.99976],
        [-1.999365095, 1.998896097, -1.999605076, -1.99892],
        [1.997775859, -1.844000202, 1.660111333, -1.37353],
        [1.818753218, -1.909687734, 1.206631506, -1.40799],
        [1.992023578, -1.923804827, 0.706593926, -1.54378],
        [1.999174385, -1.997592083, 0.21221635, -1.58697],
        [1.997854305, -1.999410881, -0.243400633, -1.82759],
        [0.851626415, 1.574201387, 1.581026838, 1.573934],
        [1.008512576, 1.570791652, 1.595657199, 1.741762],
        [1.107744254, 1.615475549, 1.723582196, 1.807615],
        [1.089897991, 1.611369928, 1.882537367, 1.873522],
        [1.290406093, 1.695289797, 1.953503509, 1.94625],
    ])
    return B


def get_matrix_T_old() -> np.ndarray:
    """
    Get baseline transition matrix T_old from Appendix 1.1.
    
    Note: Original is (5×4), but we return (4×5) = T as per our convention.
    
    Returns
    -------
    np.ndarray
        Baseline transition matrix of shape (4, 5)
    """
    # Original from appendix is shape (5, 4) = T^T
    T_transpose = np.array([
        [-0.278135369, 0.520567817, -0.140387778, 0.024426],
        [-0.382248581, 0.126035484, -0.145008015, 0.349038],
        [0.522859856, -0.341076002, 0.433255464, 0.198781],
        [-0.065904355, -0.023301678, -0.149755201, -0.25589],
        [-0.177604706, -0.49953555, -0.428847974, -0.61688],
    ])
    # Return T of shape (4, 5)
    return T_transpose.T


def get_class_labels() -> np.ndarray:
    """
    Get class labels for the 15 samples.
    
    Returns
    -------
    np.ndarray
        Class labels [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]
    """
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])


def get_synthetic_data() -> dict:
    """
    Get all synthetic data as a dictionary.
    
    Returns
    -------
    dict
        Dictionary with keys: 'A', 'B', 'T_old', 'labels', 'm', 'k', 'l'
    """
    A = get_matrix_A()
    B = get_matrix_B()
    T_old = get_matrix_T_old()
    labels = get_class_labels()
    
    m, k = A.shape
    _, l = B.shape
    
    return {
        'A': A,
        'B': B,
        'T_old': T_old,
        'labels': labels,
        'm': m,
        'k': k,
        'l': l,
    }


# Expected results from manuscript Table 1
EXPECTED_RESULTS = {
    'old_approach': {
        'mse_fid': 0.002,
        'sym_err': 1.450,
        'robustness_err': 0.850,
    },
    'new_approach': {
        'mse_fid': 0.005,
        'sym_err': 0.080,
        'robustness_err': 0.120,
    },
}
