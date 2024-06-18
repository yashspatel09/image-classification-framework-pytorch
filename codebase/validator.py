"""
Validator Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to validate a model for one epoch and calculate metrics.
"""

from tqdm import tqdm
import numpy as np
import torch
from codebase.metrics import ClassificationMetrics
from config import Config
import matplotlib.pyplot as plt

# Initialize configuration and classification metrics instances
cfg = Config()
cm = ClassificationMetrics()

def save_labelled_image(image, actual_label, predicted_label, file_path):
    """
    Save an image with actual and predicted labels as the title.

    :param image: Image tensor in CxHxW format.
    :param actual_label: Actual label of the image.
    :param predicted_label: Predicted label of the image.
    :param file_path: Path to save the labeled image.
    """
    plt.imshow(image.permute(1, 2, 0))  # assuming image tensor is in CxHxW format
    plt.title(f'Actual: {actual_label} / Predicted: {predicted_label}')
    plt.savefig(file_path)
    plt.close()

class Validator:
    def __init__(self):
        """
        Initialize the Validator.
        """
        pass

    def validate_one_epoch(self, dataloader, model, criterion, logger, epoch):
        """
        Validate the model for one epoch.

        :param dataloader: DataLoader for validation data.
        :param model: The model to be validated.
        :param criterion: Loss function.
        :param logger: Logger object for logging events and metrics.
        :param epoch: Current epoch number.
        :return: A tuple of metrics and average loss.
        """
        # Switch model to evaluation mode
        model.eval()

        # Lists to store outputs, targets, and loss
        final_targets = []
        final_outputs = []
        final_loss = []

        # Validation loop
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs, targets = batch[0].cuda(), batch[1].cuda()

            with torch.no_grad():
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Record loss
                final_loss.append(loss.item())

                # Move outputs and targets to CPU
                targets = targets.detach().cpu().numpy().tolist()
                outputs = outputs.detach().cpu().numpy().tolist()

                # Extend the lists
                final_targets.extend(targets)
                final_outputs.extend(outputs)

        # Calculate metrics
        final_outputs = np.argmax(final_outputs, axis=1)
        metrics = cm.calculate_metrics(final_targets, final_outputs)  # Calculate classification metrics

        # Calculate average loss
        average_loss = np.mean(final_loss)
        logger.log_event(epoch, metrics, 'val', average_loss)
        
        return metrics, average_loss

# Usage example
# validator = Validator()
# metrics, loss = validator.validate_one_epoch(validation_dataloader, model, criterion, logger, epoch)
