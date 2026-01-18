"""
Feature extraction for MNIST experiment.

Handles:
- Data loading (MNIST)
- FM Feature extraction (CNN penultimate layer)
- MM Feature extraction (Flattened pixels)
- Rotation augmentation for generator estimation
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional

from .model import MNIST_CNN
from src.generators import rotate_image

def get_mnist_data(root: str = './data', train: bool = True, download: bool = True) -> datasets.MNIST:
    """Get MNIST dataset."""
    # Standard transform: ToTensor (0-1) and Normalize
    # Note: Normalize((0.1307,), (0.3081,)) is standard for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=root, train=train, download=download, transform=transform)
    return dataset

def extract_features(model: MNIST_CNN, 
                     dataset: torch.utils.data.Dataset,
                     num_samples: Optional[int] = None,
                     batch_size: int = 64,
                     device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract FM (A) and MM (B) features from dataset.
    
    Parameters
    ----------
    model : MNIST_CNN
        Trained model
    dataset : Dataset
        MNIST dataset
    num_samples : int, optional
        Number of samples to use (if None, use all).
    batch_size : int
        Inference batch size
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    A : np.ndarray
        FM features (m, k)
    B : np.ndarray
        MM features (m, l) - flattened pixels of *normalized* images
    labels : np.ndarray
        Class labels (m,)
    """
    model.eval()
    
    if num_samples:
        indices = np.random.RandomState(42).choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    fm_features_list = []
    mm_features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            
            # 1. FM Features: Pass through CNN
            fm_feats = model.get_features(images)
            fm_features_list.append(fm_feats.cpu().numpy())
            
            # 2. MM Features: Flatten input images
            # images shape: (B, 1, 28, 28) -> (B, 784)
            mm_feats = images.view(images.size(0), -1)
            mm_features_list.append(mm_feats.cpu().numpy())
            
            labels_list.append(targets.numpy())
            
    A = np.vstack(fm_features_list)
    B = np.vstack(mm_features_list)
    labels = np.concatenate(labels_list)
    
    return A, B, labels

def get_rotated_features(model: MNIST_CNN,
                         dataset: torch.utils.data.Dataset,
                         indices: np.ndarray,
                         angle_rad: float,
                         device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get features for specific images rotated by angle.
    
    Used for generator estimation and robustness testing.
    
    Parameters
    ----------
    ...
    indices : np.ndarray
        Indices of samples in dataset to rotate
    angle_rad : float
        Rotation angle in radians
    
    Returns
    -------
    A_rot : np.ndarray
        FM features of rotated images
    B_rot : np.ndarray
        MM features of rotated images
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    
    fm_list = []
    mm_list = []
    
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            # images: (batch, 1, 28, 28) tensor
            imgs_np = images.numpy()
            
            # Rotate batch
            imgs_rot_np = np.zeros_like(imgs_np)
            for i in range(len(imgs_np)):
                # rotate_image expects (H,W) or (H,W,C). 
                # MNIST is (1, 28, 28) -> squeeze to (28, 28) for rotation
                img_sq = imgs_np[i].squeeze(0)
                img_rot = rotate_image(img_sq, angle_rad)
                imgs_rot_np[i] = img_rot[np.newaxis, ...]
            
            # Convert back to tensor
            images_rot = torch.from_numpy(imgs_rot_np).float().to(device)
            
            # FM
            fm = model.get_features(images_rot)
            fm_list.append(fm.cpu().numpy())
            
            # MM
            mm = images_rot.view(images_rot.size(0), -1)
            mm_list.append(mm.cpu().numpy())
            
    A_rot = np.vstack(fm_list)
    B_rot = np.vstack(mm_list)
    return A_rot, B_rot
