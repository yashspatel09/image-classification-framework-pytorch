"""
Main Training and Evaluation Script
Author: Yash Patel
Copyright: 2024
Description: This script handles the training and evaluation of a machine learning model using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from config import Config
from datasplitter import DatasetPreparer
from dataset import CustomDataset
from augment import ImageTransforms
from torch.utils.data import DataLoader
from codebase.modeltrainer import ModelTrainer
from codebase.evaluator import ModelEvaluator
from codebase.epoch_logger import EventLogger
from torchsampler import ImbalancedDatasetSampler

# Initialize configuration and logger
cfg = Config()
logger = EventLogger(cfg.filename, ['epoch', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'confusion_matrix'])
ft = ModelTrainer(logger)

# Prepare datasets
preparer = DatasetPreparer()
train_df, valid_df, test_df = preparer.get_dataframes()
print(len(train_df), len(valid_df), len(test_df))

# Set random seed for reproducibility
torch.manual_seed(cfg.seed)

# Create datasets and dataloaders
train_dataset = CustomDataset(train_df, transform=ImageTransforms(True))
valid_dataset = CustomDataset(valid_df, transform=ImageTransforms(False))

train_dataloader = DataLoader(train_dataset,
                              sampler=ImbalancedDatasetSampler(train_dataset),
                              batch_size=cfg.batch_size,
                              num_workers=6)

valid_dataloader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=6)

test_dataset = CustomDataset(test_df)
test_dataloader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=0)

# Load model
from model import Model
model = Model()

def format_parameters(num_params):
    """
    Format the number of parameters in the model for readability.

    :param num_params: Number of parameters.
    :return: Formatted string of the number of parameters.
    """
    if num_params < 1e3:
        return f"{num_params} Params"
    elif num_params < 1e6:
        return f"{num_params / 1e3:.2f}K Params"
    elif num_params < 1e9:
        return f"{num_params / 1e6:.2f}M Params"
    else:
        return f"{num_params / 1e9:.2f}B Params"

# Example usage to print number of parameters
num_params = sum(p.numel() for p in model.parameters())
formatted_params = format_parameters(num_params)
print("Number of parameters:", formatted_params)

# Utilize multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1], output_device=torch.device('cuda:0'))
model = model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()
if cfg.opt == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
elif cfg.opt == "SGD":
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay,
                                momentum=0.9)
else:
    raise ValueError("Optimizer not supported. Please choose between ADAM and SGD.")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.ceil(len(train_dataloader.dataset) / cfg.batch_size) * cfg.epochs, eta_min=cfg.lr_min)

# Train the model
acc, loss, val_acc, val_loss, model, lrs = ft.fit(model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader)

# Evaluate the model
evaluator = ModelEvaluator(model_path=cfg.loadModel, dataloader=test_dataloader, num_classes=cfg.n_classes, logger=logger)
model = evaluator.load_model(model)
metrics = evaluator.evaluate(model)
evaluator.print_metrics(metrics)
