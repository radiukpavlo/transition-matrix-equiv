"""
MNIST Data Loader Module

This module handles loading and preprocessing of the MNIST dataset.
Standard split: 60,000 training samples, 10,000 test samples.
Each image is 28×28 pixels, grayscale (1 channel).
"""

import numpy as np
from torchvision import datasets, transforms
import torch

def load_mnist(data_dir='/tmp/mnist_data'):
    """
    Load the MNIST dataset using torchvision.

    Args:
        data_dir (str): Directory to store/load MNIST data

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
            - train_data: numpy array of shape (60000, 1, 28, 28)
            - train_labels: numpy array of shape (60000,)
            - test_data: numpy array of shape (10000, 1, 28, 28)
            - test_labels: numpy array of shape (10000,)
    """
    # Define transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range and shape (1, 28, 28)
    ])

    # Load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Convert to numpy arrays
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_data = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()

    # Reshape to (N, 1, 28, 28) to explicitly include channel dimension
    train_data = train_data[:, np.newaxis, :, :]
    test_data = test_data[:, np.newaxis, :, :]

    # Normalize to [0, 1] range
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, train_labels, test_data, test_labels


def get_mnist_summary(train_data, train_labels, test_data, test_labels):
    """
    Generate a summary of the MNIST dataset.

    Args:
        train_data: Training images array
        train_labels: Training labels array
        test_data: Test images array
        test_labels: Test labels array

    Returns:
        dict: Summary statistics
    """
    summary = {
        'train_shape': train_data.shape,
        'train_labels_shape': train_labels.shape,
        'test_shape': test_data.shape,
        'test_labels_shape': test_labels.shape,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'image_shape': train_data.shape[1:],  # (1, 28, 28)
        'num_classes': len(np.unique(train_labels)),
        'train_label_distribution': {
            int(label): int(count)
            for label, count in zip(*np.unique(train_labels, return_counts=True))
        },
        'test_label_distribution': {
            int(label): int(count)
            for label, count in zip(*np.unique(test_labels, return_counts=True))
        }
    }
    return summary


if __name__ == "__main__":
    print("Loading MNIST dataset...")
    train_data, train_labels, test_data, test_labels = load_mnist()

    print("\nMNIST Dataset Summary")
    print("=" * 50)
    summary = get_mnist_summary(train_data, train_labels, test_data, test_labels)

    for key, value in summary.items():
        if key.endswith('_distribution'):
            print(f"\n{key}:")
            for label, count in value.items():
                print(f"  Class {label}: {count} samples")
        else:
            print(f"{key}: {value}")

    # Verify shapes
    assert train_data.shape == (60000, 1, 28, 28), f"Expected (60000, 1, 28, 28), got {train_data.shape}"
    assert test_data.shape == (10000, 1, 28, 28), f"Expected (10000, 1, 28, 28), got {test_data.shape}"
    print("\n✓ All shape assertions passed!")
