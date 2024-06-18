"""
ModelTrainer Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to train and validate a model, log results, and save the best model.
"""

import copy
import numpy as np
import torch
import time
from seed import SeedSetter
from .trainer import Trainer
from .validator import Validator
from config import Config
from .plot_metrics import MetricsPlotter
from tabulate import tabulate

# Initialize configuration, seed setter, trainer, and validator instances
cfg = Config()
ss = SeedSetter()
tr = Trainer()
vl = Validator()

class ModelTrainer:
    def __init__(self, logger):
        """
        Initialize the ModelTrainer with a logger.

        :param logger: Logger object for logging events and metrics.
        """
        self.best_accuracy = 0
        self.best_model_wts = None
        self.logger = logger
        self.metrics_plotter = MetricsPlotter(cfg.plotname)  # Instance of MetricsPlotter for plotting metrics

    def fit(self, model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader):
        """
        Train and validate the model.

        :param model: The model to be trained.
        :param optimizer: Optimizer for training.
        :param scheduler: Learning rate scheduler.
        :param criterion: Loss function.
        :param train_dataloader: DataLoader for training data.
        :param valid_dataloader: DataLoader for validation data.
        :return: Lists of training and validation metrics, the best model, and learning rates.
        """
        lrs = []
        acc_list, loss_list, val_acc_list, val_loss_list = [], [], [], []
        train_precision_list, val_precision_list = [], []
        train_recall_list, val_recall_list = [], []
        train_f1_list, val_f1_list = [], []
        torch.cuda.empty_cache()

        for epoch in range(cfg.epochs):
            epoch_start_time = time.time()
            print(f"\n{'=' * 30}\nEpoch {epoch + 1}/{cfg.epochs}\t Best Accuracy: {self.best_accuracy}\n{'=' * 30} ")

            # Set the seed for reproducibility
            ss.set_seed((cfg.seed + epoch))

            # Train for one epoch
            train_metrics, train_loss, lrs = tr.train_one_epoch(train_dataloader, model, optimizer, scheduler, criterion, lrs, self.logger, epoch)

            # Validate if validation data loader is provided
            if valid_dataloader:
                val_metrics, val_loss = vl.validate_one_epoch(valid_dataloader, model, criterion, self.logger, epoch)
                val_acc = val_metrics['accuracy']

                # Check for best model
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    cfg.vAcc = val_acc
                    cfg.tAcc = train_metrics['accuracy']
                    self.best_model_wts = copy.deepcopy(model.state_dict())
                    self.save_model(optimizer)

            # Append metrics to lists
            acc_list.append(train_metrics['accuracy'])
            loss_list.append(train_loss)
            train_precision_list.append(train_metrics['precision'])
            train_recall_list.append(train_metrics['recall'])
            train_f1_list.append(train_metrics['f1_score'])
            if valid_dataloader:
                val_acc_list.append(val_metrics['accuracy'])
                val_loss_list.append(val_loss)
                val_precision_list.append(val_metrics['precision'])
                val_recall_list.append(val_metrics['recall'])
                val_f1_list.append(val_metrics['f1_score'])

            epoch_duration = time.time() - epoch_start_time
            self.print_epoch_summary(epoch, train_metrics, train_loss, val_metrics, val_loss, valid_dataloader, epoch_duration)
            torch.cuda.empty_cache()

        # After the training loop, call plot_all_metrics
        self.metrics_plotter.plot_all_metrics(
            train_loss=loss_list,
            val_loss=val_loss_list,
            train_acc=acc_list,
            val_acc=val_acc_list,
            train_precision=train_precision_list,
            val_precision=val_precision_list,
            train_recall=train_recall_list,
            val_recall=val_recall_list,
            train_f1=train_f1_list,
            val_f1=val_f1_list
        )
        print("\nBest Accuracy at the end of training:", self.best_accuracy, "\n")

        return acc_list, loss_list, val_acc_list, val_loss_list, model, lrs

    def print_epoch_summary(self, epoch, train_metrics, train_loss, val_metrics, val_loss, valid_dataloader, epoch_duration):
        """
        Print the summary of the epoch.

        :param epoch: Current epoch number.
        :param train_metrics: Metrics from the training data.
        :param train_loss: Loss from the training data.
        :param val_metrics: Metrics from the validation data.
        :param val_loss: Loss from the validation data.
        :param valid_dataloader: DataLoader for validation data.
        :param epoch_duration: Duration of the epoch.
        """
        # Create a single row of data
        single_row_data = [
            f"{train_loss:.4f}",
            f"{train_metrics['accuracy']:.4f}",
            f"{train_metrics['precision']:.4f}",
            f"{train_metrics['recall']:.4f}",
            f"{train_metrics['f1_score']:.4f}"
        ]

        # Add validation data if available
        if valid_dataloader:
            single_row_data.extend([
                f"{val_loss:.4f}",
                f"{val_metrics['accuracy']:.4f}",
                f"{val_metrics['precision']:.4f}",
                f"{val_metrics['recall']:.4f}",
                f"{val_metrics['f1_score']:.4f}"
            ])

        # Create headers based on the data available
        headers = ['Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score']
        if valid_dataloader:
            headers.extend(['Validation Loss', 'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score'])

        # Print the epoch summary
        print(f"Epoch Summary for Epoch {epoch + 1}")
        print(f"Duration: {epoch_duration:.2f}s")
        print(tabulate([single_row_data], headers=headers, floatfmt=".4f"))

        # Print per-class accuracy if available
        if valid_dataloader and 'confusion_matrix' in val_metrics:
            confusion_matrix = val_metrics['confusion_matrix']
            row_sums = np.sum(confusion_matrix, axis=1)
            per_class_accuracy = np.divide(np.diag(confusion_matrix), row_sums, where=row_sums != 0)
            per_class_accuracy_labels = [f"Class {i} Acc" for i in range(len(per_class_accuracy))]
            per_class_accuracy_values = [f"{acc:.4f}" if np.isfinite(acc) else 'N/A' for acc in per_class_accuracy]

            # Print the per-class accuracy headers and values
            print(tabulate([per_class_accuracy_labels, per_class_accuracy_values], headers=[""] * len(per_class_accuracy_labels), floatfmt=".4f"))

        if valid_dataloader and 'best_accuracy' in val_metrics:
            print(f"Best Accuracy: {val_metrics['best_accuracy']:.4f}")
        print("\n")

    def save_model(self, optimizer):
        """
        Save the best model weights and optimizer state.

        :param optimizer: Optimizer used during training.
        """
        path = f"{cfg.save_model}{cfg.prefix}"
        torch.save({
            'model_state_dict': self.best_model_wts,
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
        print(f"\n{'*' * 10} New Best Model Saved! {'*' * 10}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}\nModel saved to {path}\n")

# Example usage:
# logger is a logging object that should be defined elsewhere in your code.
# model_trainer = ModelTrainer(logger)
# acc_list, loss_list, val_acc_list, val_loss_list, model, lrs = model_trainer.fit(model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader)
