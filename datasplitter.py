"""
DatasetPreparer Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to prepare datasets for training, validation, and testing.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

# Initialize configuration instance
cfg = Config()

class DatasetPreparer:
    def __init__(self):
        """
        Initialize the DatasetPreparer.
        """
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.prepare_datasets()

    def prepare_datasets(self):
        """
        Prepare the train, validation, and test dataframes based on the configuration.
        """
        print(f"Data directory: {cfg.data_dir}")  # Print the data directory
        if cfg.fixed:
            # If the dataset is fixed, load from pre-defined directories
            directories = ['train', 'val', 'test']
            for directory in directories:
                data = []
                for s, l in zip(cfg.sub_folders, cfg.labels):
                    for r, d, f in os.walk(os.path.join(cfg.data_dir, directory, s)):
                        for file in f:
                            if ".jpg" in file or ".png" in file:
                                data.append((os.path.join(directory, s, file), l))
                df = pd.DataFrame(data, columns=['file_name', 'label'])
                if directory == 'train':
                    self.train_df = df
                elif directory == 'val':
                    self.valid_df = df
                elif directory == 'test':
                    self.test_df = df
        else:
            # If the dataset is not fixed, split the dataset randomly
            data = []
            for s, l in zip(cfg.sub_folders, cfg.labels):
                dir_path = os.path.join(cfg.data_dir, s)
                print(f"Searching in: {dir_path}")  # Print the directory being searched
                for r, d, f in os.walk(dir_path):
                    for file in f:
                        if ".jpg" in file or ".png" in file:
                            data.append((os.path.join(s, file), l))
            df = pd.DataFrame(data, columns=['file_name', 'label'])
            print(f"Found {len(df)} images in total.")  # Print how many images were found
            # Split the dataset into training, validation, and test sets
            self.train_df, other_df = train_test_split(df, test_size=0.7, random_state=cfg.seed)
            self.valid_df, self.test_df = train_test_split(other_df, test_size=0.5, random_state=cfg.seed)

    def get_dataframes(self):
        """
        Return the prepared dataframes.

        :return: Tuple containing train, validation, and test dataframes.
        """
        return self.train_df, self.valid_df, self.test_df

# Example usage:
# dataset_preparer = DatasetPreparer()
# train_df, valid_df, test_df = dataset_preparer.get_dataframes()
