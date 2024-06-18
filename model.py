"""
Model Module
Author: Yash Patel
Copyright: 2024
Description: This module defines the model architecture for a classification task using a pre-trained VGG16.
"""

import torch
import torch.nn as nn
from torchvision import models
from config import Config

# Initialize configuration instance
cfg = Config()

class Model(nn.Module):
    def __init__(self):
        """
        Initialize the Model with a pre-trained VGG16 architecture.
        """
        super(Model, self).__init__()

        # Load pre-trained VGG16 model
        self.model = models.vgg16(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, cfg.n_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = self.model(x)
        return x

# Use this part to test the model if needed
if __name__ == "__main__":
    test_model = Model()
    print("Number of parameters: ", sum(p.numel() for p in test_model.parameters()))
    test_input = torch.randn(1, 3, 256, 256)  # Create a random input tensor
    test_output = test_model(test_input)
    print(test_output.size())
