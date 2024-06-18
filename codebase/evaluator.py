"""
ModelEvaluator Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to evaluate a trained model on a dataset and log the results.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from codebase.metrics import ClassificationMetrics
from codebase.plot_metrics import MetricsPlotter  # Make sure this import is correct based on your file structure
from config import Config
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image

# Initialize configuration and metric plotting instances
cfg = Config()
cm = ClassificationMetrics()
metrics_plotter = MetricsPlotter(cfg.plotname)

class ModelEvaluator:
    def __init__(self, model_path, dataloader, num_classes, logger):
        """
        Initialize the ModelEvaluator with model path, dataloader, number of classes, and logger.

        :param model_path: Path to the trained model file.
        :param dataloader: DataLoader object for the dataset to evaluate.
        :param num_classes: Number of classes in the dataset.
        :param logger: Logger object for logging events and metrics.
        """
        self.model_path = model_path
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.logger = logger
        self.metrics_plotter = metrics_plotter  # Instance of MetricsPlotter for plotting metrics
        self.all_images = []
        self.all_actual_labels = []
        self.all_predicted_labels = []

    def load_model(self, model):
        """
        Load the model from the specified path and prepare it for evaluation.

        :param model: The model architecture to be loaded with trained weights.
        :return: The loaded model ready for evaluation.
        """
        # Check if multiple GPUs are available and use DataParallel if so
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(self.model_path)['model_state_dict'])  # Load model weights
        model.cuda()  # Move the model to GPU
        model.eval()  # Set the model to evaluation mode
        return model

    def evaluate(self, model):
        """
        Evaluate the model on the provided dataset and log the metrics.

        :param model: The loaded model to be evaluated.
        :return: A dictionary of calculated metrics.
        """
        final_y = []
        final_y_pred = []
        final_y_score = []  # For ROC and AUC

        # Iterate over the dataset using the dataloader
        for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            X, y = batch[0].cuda(), batch[1].cuda()  # Move batch data to GPU
            with torch.no_grad():  # Disable gradient calculation for evaluation
                y_score = model(X)  # Get model predictions
                y_pred = torch.argmax(y_score, dim=1)  # Get predicted class
                y = y.detach().cpu().numpy()  # Move ground truth labels to CPU
                y_pred = y_pred.detach().cpu().numpy()  # Move predicted labels to CPU
                y_score = y_score.detach().cpu().numpy()  # Move class probabilities to CPU
                final_y.extend(y)  # Append ground truth labels
                final_y_pred.extend(y_pred)  # Append predicted labels
                final_y_score.extend(y_score)  # Append class probabilities
                self.accumulate_batch_results(X.cpu(), y, y_pred)  # Accumulate batch results

        # Convert lists to numpy arrays
        final_y = np.array(final_y)
        final_y_pred = np.array(final_y_pred)
        final_y_score = np.array(final_y_score)

        metrics = cm.calculate_metrics(final_y, final_y_pred)  # Calculate classification metrics
        self.logger.log_event('999', metrics, 'test')  # Log the metrics

        self.create_single_snapshot(cfg.batch_size)  # Create a snapshot of results

        # Plot the ROC curve for each class
        self.metrics_plotter.plot_multi_class_roc_curve(final_y, final_y_score, cfg.n_classes, title='Test ROC Curve')

        # Plot the AUC curve for each class
        self.metrics_plotter.plot_multi_class_auc_curve(final_y, final_y_score, cfg.n_classes, title='Test AUC Curve')

        # Plot the AUC bar chart for each class
        self.metrics_plotter.plot_multi_class_auc_bar(final_y, final_y_score, cfg.n_classes, title='Test AUC Bar Chart')

        # Plot Confusion Matrix
        self.metrics_plotter.plot_confusion_matrix(final_y, final_y_pred, classes=cfg.sub_folders, title='Test Confusion Matrix')

        return metrics

    def print_metrics(self, metrics):
        """
        Print the calculated metrics in a tabular format.

        :param metrics: Dictionary containing calculated metrics.
        """
        # General metrics
        metric_data = [
            ['Accuracy', metrics['accuracy']],
            ['Precision', metrics['precision']],
            ['Recall', metrics['recall']],
            ['F1 Score', metrics['f1_score']]
        ]
        metric_table = tabulate(metric_data, headers=['Metric', 'Value'], tablefmt='grid', floatfmt=".4f")
        print("\nTest Metrics:\n", metric_table)

        # Per-class accuracy
        confusion_matrix = metrics['confusion_matrix']
        row_sums = np.sum(confusion_matrix, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_accuracy = np.diag(confusion_matrix) / row_sums
            per_class_accuracy = np.nan_to_num(per_class_accuracy)  # Convert NaNs to 0

        per_class_accuracy_data = [[f"Class {i}", f"{acc:.4f}" if row_sums[i] > 0 else "N/A"] for i, acc in enumerate(per_class_accuracy)]
        per_class_accuracy_table = tabulate(per_class_accuracy_data, headers=['Class', 'Accuracy'], tablefmt='grid', floatfmt=".4f")
        print("\nPer-Class Accuracy:\n", per_class_accuracy_table)

        print("\nTest Confusion Matrix:\n", confusion_matrix, "\n")

    def accumulate_batch_results(self, images, actual_labels, predicted_labels):
        """
        Accumulate the results from each batch.

        :param images: List of image tensors.
        :param actual_labels: List of actual labels.
        :param predicted_labels: List of predicted labels.
        """
        self.all_images.extend(images)  # Append images
        self.all_actual_labels.extend(actual_labels)  # Append actual labels
        self.all_predicted_labels.extend(predicted_labels)  # Append predicted labels

    def add_border(self, img_tensor, border_color):
        """
        Add a border to the image tensor.

        :param img_tensor: Image tensor.
        :param border_color: Color of the border.
        :return: PIL image with the border.
        """
        # Convert tensor to numpy array and then to PIL image
        np_img = img_tensor.numpy().transpose(1, 2, 0)
        img = Image.fromarray((np_img * 255).astype('uint8'))
        bordered_img = Image.new('RGB', (img.width + 10, img.height + 10), border_color)
        bordered_img.paste(img, (5, 5))
        return bordered_img

    def create_single_snapshot(self, batch_size, snapshot_filename='snapshot.png'):
        """
        Create a snapshot of the accumulated results.

        :param batch_size: Number of images per batch.
        :param snapshot_filename: Filename for the snapshot image.
        """
        total_images = len(self.all_images)
        num_full_batches = total_images // batch_size
        leftover_images = total_images % batch_size
        num_batches = num_full_batches + (1 if leftover_images else 0)

        # Adjust the subplots layout based on the number of batches and images per batch.
        fig, axes = plt.subplots(num_batches, batch_size, figsize=(batch_size * 2, num_batches * 2), squeeze=False)

        for i, (image, actual_label, predicted_label) in enumerate(zip(self.all_images, self.all_actual_labels, self.all_predicted_labels)):
            batch_idx = i // batch_size
            img_idx = i % batch_size
            border_color = 'green' if actual_label == predicted_label else 'red'
            bordered_img = self.add_border(image, border_color)

            ax = axes[batch_idx, img_idx]
            ax.imshow(bordered_img)
            ax.set_title(f'Actual: {actual_label}\nPredicted: {predicted_label}')
            ax.axis('off')

            # Hide any unused axes if the last batch is not full
            if batch_idx == num_batches - 1 and leftover_images and img_idx == leftover_images - 1:
                for j in range(img_idx + 1, batch_size):
                    axes[batch_idx, j].axis('off')

        plt.tight_layout()
        plt.savefig(cfg.plotname + snapshot_filename, bbox_inches='tight')
        plt.close()

# Usage example:
# model_evaluator = ModelEvaluator(model_path, dataloader, num_classes, logger, metrics_plotter)
# model = SomeModel()  # Replace with your actual model
# model = model_evaluator.load_model(model)
# test_metrics = model_evaluator.evaluate(model)
# model_evaluator.print_metrics(test_metrics)
