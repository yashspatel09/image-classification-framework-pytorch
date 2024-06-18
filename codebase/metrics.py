"""
ClassificationMetrics Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to calculate classification metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from config import Config

cfg = Config()

class ClassificationMetrics:
    def __init__(self, average_method='macro'):
        """
        Initialize the ClassificationMetrics with the specified average method.

        :param average_method: Method to calculate average for precision, recall, and f1 score.
        """
        self.labels = cfg.sub_folders  # List of class labels
        self.average_method = average_method  # Averaging method for metrics

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various classification metrics based on true and predicted labels.

        :param y_true: Array of true labels.
        :param y_pred: Array of predicted labels.
        :return: Dictionary containing calculated metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),  # Calculate accuracy
            'confusion_matrix': confusion_matrix(y_true, y_pred),  # Calculate confusion matrix
            'precision': precision_score(y_true, y_pred, average=self.average_method, zero_division=0),  # Calculate precision
            'recall': recall_score(y_true, y_pred, average=self.average_method, zero_division=0),  # Calculate recall
            'f1_score': f1_score(y_true, y_pred, average=self.average_method, zero_division=0)  # Calculate f1 score
        }

        # Calculate ROC AUC score if binary classification
        if y_pred is not None and len(self.labels) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)

        return metrics

    def get_metrics(self):
        """
        Return the calculated metrics.

        :return: Dictionary containing calculated metrics.
        """
        return self.metrics
