"""
Trainer Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to train a model for one epoch and calculate metrics.
"""

from tqdm import tqdm
import numpy as np
from config import Config
import torch
from codebase.metrics import ClassificationMetrics

# Initialize configuration and classification metrics instances
cfg = Config()
cm = ClassificationMetrics()

class Trainer:
    def __init__(self):
        """
        Initialize the Trainer.
        """
        pass

    def train_one_epoch(self, dataloader, model, optimizer, scheduler, criterion, lrs, logger, epoch):
        """
        Train the model for one epoch.

        :param dataloader: DataLoader for training data.
        :param model: The model to be trained.
        :param optimizer: Optimizer for training.
        :param scheduler: Learning rate scheduler.
        :param criterion: Loss function.
        :param lrs: List to store learning rates.
        :param logger: Logger object for logging events and metrics.
        :param epoch: Current epoch number.
        :return: A tuple of metrics, average loss, and learning rates.
        """
        # Switch model to training mode
        model.train()

        # Lists for metrics and learning rates
        final_y = []
        final_y_pred = []
        final_loss = []
        lrs = []

        # Training loop
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y = batch[0].cuda(), batch[1].cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                y_pred = model(x)
                
                # Compute loss
                loss = criterion(y_pred, y)

                # Record loss and learning rate
                final_loss.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

                # Convert predictions and targets to CPU and flatten
                y = y.detach().cpu().numpy().tolist()
                y_pred = y_pred.detach().cpu().numpy().tolist()

                # Extend the lists
                final_y.extend(y)
                final_y_pred.extend(y_pred)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Step the scheduler
            scheduler.step()

        # Calculate metrics
        final_y_pred = np.argmax(final_y_pred, axis=1)
        metric = cm.calculate_metrics(final_y, final_y_pred)  # Calculate classification metrics
        
        # Calculate average loss
        average_loss = np.mean(final_loss)
        logger.log_event(epoch, metric, 'train', average_loss)
        
        return metric, average_loss, lrs

# Usage example
# trainer = Trainer()
# metric, loss, lrs = trainer.train_one_epoch(dataloader, model, optimizer, scheduler, criterion, lrs, logger, epoch)
