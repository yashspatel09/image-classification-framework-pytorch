"""
SeedSetter Module
Author: Yash Patel
Copyright: 2024
Description: This module sets seeds for reproducibility in a machine learning project.
"""

import random
import os
import numpy as np
import torch
from config import Config

# Initialize configuration instance
cfg = Config()

class SeedSetter:
    def __init__(self):
        """
        Initialize the SeedSetter and set the seed from configuration.
        """
        self.set_seed(cfg.seed)  # Set seed from configuration

    def set_seed(self, seed):
        """
        Set the seed for various random number generators for reproducibility.

        :param seed: Seed value to set.
        """
        random.seed(seed)  # Set the seed for the random module
        os.environ["PYTHONHASHSEED"] = str(seed)  # Set the PYTHONHASHSEED environment variable
        np.random.seed(seed)  # Set the seed for numpy
        torch.manual_seed(seed)  # Set the seed for PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # Set the seed for CUDA on the current GPU
            torch.cuda.manual_seed_all(seed)  # Set the seed for CUDA on all available GPUs
        
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking

# Example usage:
# seed_setter = SeedSetter()
