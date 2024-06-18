"""
CustomDataset Module
Author: Yash Patel
Copyright: 2024
Description: This module provides a custom dataset class for loading and transforming images and their labels.
"""

import os
import cv2
from torch.utils.data import Dataset
from augment import ImageTransforms
from config import Config

# Initialize configuration instance
cfg = Config()

class CustomDataset(Dataset):
    def __init__(self, df, transform=False):
        """
        Initialize the CustomDataset with a dataframe and transformations.

        :param df: DataFrame containing file names and labels.
        :param transform: Boolean indicating whether to apply transformations (True for training, False for validation).
        """
        self.data_dir = cfg.data_dir  # Directory where data is stored
        self.df = df  # DataFrame containing file names and labels
        self.file_names = df['file_name'].values  # Extract file names from the DataFrame
        self.labels = df['label'].values  # Extract labels from the DataFrame
        if transform:
            self.transform = ImageTransforms(train=True)  # Apply training transformations
        else:
            self.transform = ImageTransforms(train=False)  # Apply validation transformations

    def __len__(self):
        """
        Return the number of items in the dataset.

        :return: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve an item by its index.

        :param idx: Index of the item to retrieve.
        :return: A tuple containing the transformed image and its label.
        """
        label = self.labels[idx]  # Get the label of the image
        file_path = os.path.join(self.data_dir, self.file_names[idx])  # Get the file path of the image

        # Read an image with OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f'The image file {file_path} was not found.')

        # Convert the image to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations if any
        if self.transform:
            image = self.transform(image=image)['image']

        image = image / 255.0  # Normalize to [0, 1]

        return image, label

    def get_labels(self):
        """
        Return the labels for all data points in the dataset.

        :return: Array of labels.
        """
        return self.labels

# Example usage:
# df = some_pandas_dataframe  # The DataFrame should contain 'file_name' and 'label' columns
# dataset = CustomDataset(df, transform=True)
