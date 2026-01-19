"""
MNIST CNN Model
Implements the CNN architecture from Appendix B (Table A1) of the manuscript.
"""

import torch
import torch.nn as nn


class MNISTCNN(nn.Module):
    """
    CNN model for MNIST digit classification.

    Architecture based on Table A1 in Appendix B:
    - conv_block_1: Conv2d(1->10, 3x3, stride=1, padding=1) + ReLU + MaxPool2d(2x2, stride=2)
    - conv_block_2: Conv2d(10->10, 3x3, stride=1, padding=1) + ReLU + MaxPool2d(2x2, stride=2)
    - Flatten: produces 7x7x10 = 490 features
    - Linear: 490 -> 10 (classifier)

    The penultimate layer has 490 features, which will be used as the Formal Model (A matrix).
    """

    def __init__(self):
        super(MNISTCNN, self).__init__()

        # conv_block_1: Input 28x28x1 -> Output 14x14x10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv_block_2: Input 14x14x10 -> Output 7x7x10
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10,
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier block
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(in_features=490, out_features=10, bias=True)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, 10)
        """
        # conv_block_1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # conv_block_2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def extract_features(self, x):
        """
        Extract penultimate layer features (490-dimensional vectors).

        This method extracts features from the penultimate layer,
        which will be used as the Formal Model representation (matrix A).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28)

        Returns
        -------
        torch.Tensor
            Feature vectors of shape (batch_size, 490)
        """
        # conv_block_1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # conv_block_2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten to get 490 features (penultimate layer)
        x = self.flatten(x)

        return x

    def get_penultimate_features(self, x):
        """Alias for extract_features for compatibility with generators.py."""
        return self.extract_features(x)


def verify_architecture():
    """
    Verify the CNN architecture matches the specifications.

    Expected:
    - Input: (batch_size, 1, 28, 28)
    - After conv_block_1: (batch_size, 10, 14, 14)
    - After conv_block_2: (batch_size, 10, 7, 7)
    - After flatten: (batch_size, 490)
    - Output: (batch_size, 10)
    """
    model = MNISTCNN()

    # Create a dummy input
    dummy_input = torch.randn(1, 1, 28, 28)

    print("=" * 60)
    print("CNN Architecture Verification")
    print("=" * 60)
    print(f"Input shape: {dummy_input.shape}")

    # Test forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Penultimate layer features shape: {features.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    # Verify dimensions
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    assert features.shape == (1, 490), f"Expected features shape (1, 490), got {features.shape}"

    print("\n✓ Architecture verification passed!")

    return model


if __name__ == "__main__":
    verify_architecture()
