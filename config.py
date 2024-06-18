"""
Configuration Module
Author: Yash Patel
Copyright: 2024
Description: This module provides configuration settings for a machine learning project, including directory setups and saving configurations.

""" 

import numpy as np
import torch
import torch.nn as nn
import os
from types import SimpleNamespace
import shutil

class Config:
    def __init__(self):
        """
        Initialize the Config class with default settings.
        """
        self.fixed = True
        self.batch_size = 16
        self.image_size = 256
        self.backbone = 'custom'
        self.pretrained = True
        self.weight_decay = 0
        self.learning_rate = 1e-4
        self.lr_min = 1e-7
        self.epochs = 100
        self.seed = 42
        self.opt = "ADAM"

        self.data_set_type = 'Data'  # Dataset Folder Name
        self.model_file_name = 'model.py'

        self.data_dir = f'{self.data_set_type}/'
        self.base_dir = f'customNet_Breast_{self.data_set_type}/'
        self.sub_fl = [f.path for f in os.scandir(self.data_dir + "/test") if f.is_dir()]
        self.sub_folders = [os.path.basename(subfolder) for subfolder in self.sub_fl]
        self.labels = [i for i in range(len(self.sub_folders))]
        self.n_classes = len(self.sub_folders)

        # Create necessary directories
        self.create_directory(self.base_dir)
        self.save_dir = os.path.join(self.base_dir, 'imgs/')
        self.create_directory(self.save_dir)
        self.save_model = os.path.join(self.base_dir, 'models/')
        self.create_directory(self.save_model)
        self.save_excels = os.path.join(self.base_dir, 'excels/')
        self.create_directory(self.save_excels)
        self.saveCfg = os.path.join(self.base_dir, 'config/')
        self.create_directory(self.saveCfg)
        self.savemod = os.path.join(self.base_dir, 'model_file/')
        self.create_directory(self.savemod)
        
        self.prefix = f"{self.n_classes}_wound_{self.backbone}_model.pth"
        self.loadModel = os.path.join(self.save_model, self.prefix)
        self.filename = os.path.join(self.save_excels, f'{self.prefix}.csv')
        self.plotname = self.save_dir

        self.save_to_file(self.saveCfg)
        self.move_model_file(self.model_file_name, self.savemod)

    def as_namespace(self):
        """
        Return the configuration as a SimpleNamespace object for attribute-like access.

        :return: SimpleNamespace object containing configuration attributes.
        """
        return SimpleNamespace(**self.__dict__)

    def save_to_file(self, file_path):
        """
        Save the configuration parameters to a file.

        :param file_path: Path to save the configuration file.
        """
        with open(os.path.join(file_path, "cfg.txt"), 'w') as file:
            for attr, value in self.__dict__.items():
                if attr != 'file_path':  # Skip the file_path attribute to only save configuration parameters
                    file.write(f'{attr}: {value}\n')

    def move_model_file(self, model_file_name, savemod):
        """
        Move the model file to the specified directory.

        :param model_file_name: Name of the model file to move.
        :param savemod: Directory to move the model file to.
        """
        shutil.copy(model_file_name, savemod)

    @staticmethod
    def create_directory(directory_path):
        """
        Create a directory if it does not exist.

        :param directory_path: Path of the directory to create.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

# Example usage:
cfg = Config()
cfg_namespace = cfg.as_namespace()
# print(cfg_namespace)
