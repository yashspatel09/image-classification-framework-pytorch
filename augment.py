"""
ImageTransforms Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to apply image transformations for training and validation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

# Initialize configuration instance
cfg = Config()

class ImageTransforms:
    def __init__(self, train=False):
        """
        Initialize the ImageTransforms with a set of transformations for training or validation.

        :param train: Boolean indicating whether the transformations are for training (True) or validation (False).
        """
        image_size = cfg.image_size  # Get the image size from configuration
        if train:
            # Define transformations for training
            self.transform_soft = A.Compose([
                A.Resize(image_size, image_size),  # Resize image
                A.Rotate(p=0.6, limit=[-45, 45]),  # Random rotation
                A.VerticalFlip(p=0.6),  # Random vertical flip
                A.HorizontalFlip(p=0.6),  # Random horizontal flip
                A.OneOf([
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=1, border_mode=0),  # Scale only
                    A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=1, border_mode=0),   # Rotate only
                    A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),  # Shift only
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),  # Affine transform
                ], p=0.6),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),  # Add Gaussian noise
                    A.GaussNoise(),  # Add Gauss noise
                ], p=0.4),
                A.CoarseDropout(max_holes=1, max_height=64, max_width=64, p=0.3),  # Coarse dropout
                ToTensorV2()  # Convert image to tensor
            ])
        else:
            # Define transformations for validation
            self.transform_soft = A.Compose([
                A.Resize(image_size, image_size),  # Resize image
                ToTensorV2()  # Convert image to tensor
            ])

    def __call__(self, image):
        """
        Apply the transformations to the given image.

        :param image: Image to be transformed.
        :return: Transformed image.
        """
        return self.transform_soft(image=image)

# Example usage:
# train_transforms = ImageTransforms(train=True)
# transformed_image = train_transforms(image)
