"""
MNIST CNN Model definition.

Architecture references the description in [4] and Section 3.5:
- Convolutional layers for feature extraction
- Fully connected layers
- Penultimate layer with k=490 units (FM features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self, k_features: int = 490):
        super(MNIST_CNN, self).__init__()
        # Standard simple CNN architecture for MNIST
        # Conv1: 1 -> 32 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv2: 32 -> 64 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Feature sizes after pooling:
        # Input: 28x28
        # After conv1 + pool: 14x14
        # After conv2 + pool: 7x7
        # Flattened size: 64 * 7 * 7 = 3136
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        
        # Penultimate layer (Feature Space for our analysis)
        self.fc_features = nn.Linear(512, k_features)
        
        # Output layer (10 digits)
        self.fc_out = nn.Linear(k_features, 10)
        
    def forward(self, x):
        """Standard forward pass returning class logits."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_features(x)) # Features are here (relu or not? Manuscript implies features A)
        # Usually features are taken *after* activation if they are to be compared linearly
        # but let's assume raw values or relu. Let's stick with ReLU as standard.
        x = self.fc_out(x)
        return x

    def get_features(self, x):
        """Extract penultimate layer features (FM space)."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        # Return features from the k=490 layer
        features = F.relu(self.fc_features(x))
        return features

def load_trained_model(path: str = None, device='cpu'):
    """Load model weights (placeholder since we might train from scratch)."""
    model = MNIST_CNN()
    model.to(device)
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    return model
