"""
Baseline Reproduction (Old Approach)

This script implements the baseline approach from the manuscript:
1. Train CNN on MNIST dataset
2. Extract Formal Model features (A) from CNN penultimate layer
3. Extract Mental Model features (B) from flattened images
4. Compute transition matrix T_old using training data
5. Evaluate reconstruction on test data
6. Save metrics and artifacts
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import local modules
from model import MNISTCNN
from baseline_utils import compute_transition_matrix, reconstruct_mental_features, calculate_metrics
from mnist_loader import load_mnist


def train_cnn(model, train_loader, device, epochs=10, learning_rate=0.001):
    """
    Train the CNN model on MNIST dataset.

    Parameters
    ----------
    model : MNISTCNN
        The CNN model to train
    train_loader : DataLoader
        Training data loader
    device : torch.device
        Device to train on (CPU or CUDA)
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer

    Returns
    -------
    model : MNISTCNN
        Trained model
    train_history : dict
        Training history (loss and accuracy per epoch)
    """
    print("=" * 80)
    print("Training CNN on MNIST Dataset")
    print("=" * 80)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}s")

        # Epoch statistics
        avg_loss = epoch_loss / batch_count
        accuracy = 100.0 * correct / total
        train_history['loss'].append(avg_loss)
        train_history['accuracy'].append(accuracy)

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        print("-" * 80)

    print(f"\n✓ Training completed!")
    print(f"  Final Accuracy: {train_history['accuracy'][-1]:.2f}%")
    print("=" * 80)

    return model, train_history


def evaluate_cnn(model, test_loader, device):
    """
    Evaluate the CNN model on test dataset.

    Parameters
    ----------
    model : MNISTCNN
        Trained CNN model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to evaluate on

    Returns
    -------
    accuracy : float
        Test accuracy (percentage)
    """
    print("\n" + "=" * 80)
    print("Evaluating CNN on Test Dataset")
    print("=" * 80)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx+1}/{len(test_loader)} batches...")

    accuracy = 100.0 * correct / total
    print(f"\n✓ Evaluation completed!")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print("=" * 80)

    return accuracy


def extract_features(model, data_loader, device):
    """
    Extract penultimate layer features from CNN.

    Parameters
    ----------
    model : MNISTCNN
        Trained CNN model
    data_loader : DataLoader
        Data loader
    device : torch.device
        Device to extract on

    Returns
    -------
    features : np.ndarray
        Extracted features, shape (n_samples, 490)
    labels : np.ndarray
        Corresponding labels, shape (n_samples,)
    """
    print("\n" + "=" * 80)
    print("Extracting Penultimate Layer Features (Formal Model A)")
    print("=" * 80)

    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(data_loader):
            images = images.to(device)

            # Extract features (490-dimensional)
            batch_features = model.extract_features(images)
            features_list.append(batch_features.cpu().numpy())
            labels_list.append(batch_labels.numpy())

            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx+1}/{len(data_loader)} batches...")

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    print(f"\n✓ Feature extraction completed!")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print("=" * 80)

    return features, labels


def flatten_images(images):
    """
    Flatten images to create Mental Model features (B matrix).

    Parameters
    ----------
    images : np.ndarray
        Images, shape (n_samples, 1, 28, 28) or (n_samples, 28, 28)

    Returns
    -------
    flattened : np.ndarray
        Flattened images, shape (n_samples, 784)
    """
    print("\n" + "=" * 80)
    print("Creating Mental Model Features (Flattened Images B)")
    print("=" * 80)

    if len(images.shape) == 4:
        # Shape: (n_samples, 1, 28, 28) -> (n_samples, 784)
        flattened = images.reshape(images.shape[0], -1)
    elif len(images.shape) == 3:
        # Shape: (n_samples, 28, 28) -> (n_samples, 784)
        flattened = images.reshape(images.shape[0], -1)
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    print(f"  Input shape: {images.shape}")
    print(f"  Flattened shape: {flattened.shape}")
    print("=" * 80)

    return flattened


def main():
    """
    Main execution function for baseline reproduction.
    """
    print("\n" + "=" * 80)
    print("BASELINE REPRODUCTION (OLD APPROACH)")
    print("=" * 80)
    print("Objective: Train CNN, compute T_old, and evaluate reconstruction")
    print("=" * 80 + "\n")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create output directories
    os.makedirs('workflow/data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # =========================================================================
    # Step 1: Load MNIST Data
    # =========================================================================
    print("=" * 80)
    print("Step 1: Loading MNIST Data")
    print("=" * 80)

    train_data, train_labels, test_data, test_labels = load_mnist()

    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_data)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_images = torch.FloatTensor(test_data)
    test_labels_tensor = torch.LongTensor(test_labels)

    # Create DataLoaders
    batch_size = 128
    train_dataset = TensorDataset(train_images, train_labels_tensor)
    test_dataset = TensorDataset(test_images, test_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("=" * 80 + "\n")

    # =========================================================================
    # Step 2: Train CNN Model
    # =========================================================================
    print("=" * 80)
    print("Step 2: Training CNN Model")
    print("=" * 80)

    model = MNISTCNN()
    print(f"Model architecture:\n{model}\n")

    # Train for 10 epochs (target >98% accuracy)
    model, train_history = train_cnn(
        model, train_loader, device,
        epochs=10, learning_rate=0.001
    )

    # Evaluate on test set
    test_accuracy = evaluate_cnn(model, test_loader, device)

    # Save trained model
    model_path = 'workflow/data/cnn_mnist.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Check if target accuracy achieved
    if test_accuracy >= 98.0:
        print(f"✓ Target accuracy (>98%) achieved: {test_accuracy:.2f}%")
    else:
        print(f"⚠ Target accuracy (>98%) not achieved: {test_accuracy:.2f}%")
        print(f"  Note: Continuing with analysis...")

    # =========================================================================
    # Step 3: Extract Features (Formal Model A)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Extracting Formal Model Features (A)")
    print("=" * 80)

    A_train, train_labels_extracted = extract_features(model, train_loader, device)
    A_test, test_labels_extracted = extract_features(model, test_loader, device)

    print(f"\nA_train shape: {A_train.shape}")
    print(f"A_test shape: {A_test.shape}")

    # =========================================================================
    # Step 4: Create Mental Model Features (B)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Creating Mental Model Features (B)")
    print("=" * 80)

    B_train = flatten_images(train_data)
    B_test = flatten_images(test_data)

    print(f"\nB_train shape: {B_train.shape}")
    print(f"B_test shape: {B_test.shape}")

    # =========================================================================
    # Step 5: Compute Transition Matrix T_old
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Computing Transition Matrix T_old")
    print("=" * 80)
    print("Using training data (A_train, B_train)")

    T_old = compute_transition_matrix(A_train, B_train)

    # Save T_old
    t_old_path = 'workflow/data/t_old_mnist.npy'
    np.save(t_old_path, T_old)
    print(f"\n✓ T_old saved to: {t_old_path}")
    print(f"  T_old shape: {T_old.shape}")

    # =========================================================================
    # Step 6: Evaluate Reconstruction on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating Reconstruction on Test Set")
    print("=" * 80)

    # Reconstruct test features: B*_test = A_test T_old^T
    B_test_reconstructed = reconstruct_mental_features(A_test, T_old)

    print(f"\nOriginal B_test shape: {B_test.shape}")
    print(f"Reconstructed B*_test shape: {B_test_reconstructed.shape}")

    # =========================================================================
    # Step 7: Calculate Metrics
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Calculating Reconstruction Metrics")
    print("=" * 80)

    metrics = calculate_metrics(B_test, B_test_reconstructed)

    # Save metrics
    metrics_path = 'results/baseline_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics saved to: {metrics_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINE REPRODUCTION SUMMARY")
    print("=" * 80)
    print(f"CNN Test Accuracy: {test_accuracy:.2f}%")
    print(f"T_old shape: {T_old.shape}")
    print(f"\nReconstruction Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  SSIM: {metrics['ssim_mean']:.6f} ± {metrics['ssim_std']:.6f}")
    print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print("\nOutput Files:")
    print(f"  - {model_path}")
    print(f"  - {t_old_path}")
    print(f"  - {metrics_path}")
    print("=" * 80)

    print("\n✓ Baseline reproduction completed successfully!")

    return model, T_old, metrics, test_accuracy


if __name__ == "__main__":
    main()
