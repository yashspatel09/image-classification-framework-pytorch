"""
MetricsPlotter Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to plot various evaluation metrics for classification models.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pandas as pd
import seaborn as sns
import numpy as np
import os

class MetricsPlotter:
    def __init__(self, save_path):
        """
        Initialize the MetricsPlotter with a path to save the plots.

        :param save_path: Path to save the plots.
        """
        self.save_path = save_path  # Path to save the plots
        # Create the directory if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def plot_multi_class_roc_curve(self, y_true, y_score, n_classes, title='Multi-Class ROC', file_name='multi_class_roc.png'):
        """
        Plot the ROC curve for multi-class classification.

        :param y_true: True labels.
        :param y_score: Predicted scores.
        :param n_classes: Number of classes.
        :param title: Title of the plot.
        :param file_name: Filename to save the plot.
        """
        # Binarize the output labels for each class
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_path, file_name))
        plt.close()

    def plot_multi_class_auc_curve(self, y_true, y_score, n_classes, title='Multi-Class AUC', file_name='multi_class_auc.png'):
        """
        Plot the AUC curve for multi-class classification.

        :param y_true: True labels.
        :param y_score: Predicted scores.
        :param n_classes: Number of classes.
        :param title: Title of the plot.
        :param file_name: Filename to save the plot.
        """
        # Binarize the output labels for each class
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_path, file_name))
        plt.close()
    
    def plot_multi_class_auc_bar(self, y_true, y_score, n_classes, title='Multi-Class AUC', file_name='multi_class_auc_bar.png'):
        """
        Plot the AUC values as a bar chart for multi-class classification.

        :param y_true: True labels.
        :param y_score: Predicted scores.
        :param n_classes: Number of classes.
        :param title: Title of the plot.
        :param file_name: Filename to save the plot.
        """
        # Binarize the output labels for each class
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        roc_auc = dict()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr, tpr)

        # Plot AUC values as a bar chart
        plt.figure()
        auc_values = [roc_auc[i] for i in range(n_classes)]
        plt.bar(range(n_classes), auc_values, color='skyblue')
        plt.xticks(range(n_classes), [f'Class {i}' for i in range(n_classes)])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Class')
        plt.ylabel('AUC Value')
        for i, v in enumerate(auc_values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
        plt.title(title)
        plt.savefig(os.path.join(self.save_path, file_name))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title='Confusion matrix', file_name='confusion_matrix.png'):
        """
        Plot the confusion matrix.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param classes: List of class names.
        :param normalize: Whether to normalize the confusion matrix.
        :param title: Title of the plot.
        :param file_name: Filename to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt="g" if normalize else 'd', xticklabels=classes, yticklabels=classes, cmap=plt.cm.Blues)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)
        plt.savefig(os.path.join(self.save_path, file_name))
        plt.close()

    def _plot_metrics(self, train_values, val_values, title, ylabel, xlabel, metric_name):
        """
        Helper function to plot training and validation metrics.

        :param train_values: List of training values.
        :param val_values: List of validation values.
        :param title: Title of the plot.
        :param ylabel: Label for the Y-axis.
        :param xlabel: Label for the X-axis.
        :param metric_name: Metric name for the filename.
        """
        epochs = range(1, len(train_values) + 1)
        plt.plot(epochs, train_values, 'bo-', label='Training')
        plt.plot(epochs, val_values, 'ro-', label='Validation')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(self.save_path, f"{metric_name}_plot.png"))
        plt.close()

    def plot_loss(self, train_loss, val_loss):
        """
        Plot training and validation loss.

        :param train_loss: List of training loss values.
        :param val_loss: List of validation loss values.
        """
        self._plot_metrics(train_loss, val_loss, 'Training and Validation Loss', 'Loss', 'Epoch', 'loss')

    def plot_accuracy(self, train_acc, val_acc):
        """
        Plot training and validation accuracy.

        :param train_acc: List of training accuracy values.
        :param val_acc: List of validation accuracy values.
        """
        self._plot_metrics(train_acc, val_acc, 'Training and Validation Accuracy', 'Accuracy', 'Epoch', 'accuracy')

    def plot_precision(self, train_precision, val_precision):
        """
        Plot training and validation precision.

        :param train_precision: List of training precision values.
        :param val_precision: List of validation precision values.
        """
        self._plot_metrics(train_precision, val_precision, 'Training and Validation Precision', 'Precision', 'Epoch', 'precision')

    def plot_recall(self, train_recall, val_recall):
        """
        Plot training and validation recall.

        :param train_recall: List of training recall values.
        :param val_recall: List of validation recall values.
        """
        self._plot_metrics(train_recall, val_recall, 'Training and Validation Recall', 'Recall', 'Epoch', 'recall')

    def plot_f1(self, train_f1, val_f1):
        """
        Plot training and validation F1 score.

        :param train_f1: List of training F1 score values.
        :param val_f1: List of validation F1 score values.
        """
        self._plot_metrics(train_f1, val_f1, 'Training and Validation F1 Score', 'F1 Score', 'Epoch', 'f1')

    def plot_all_metrics(self, 
                         train_loss, val_loss, 
                         train_acc, val_acc, 
                         train_precision, val_precision, 
                         train_recall, val_recall, 
                         train_f1, val_f1):
        """
        Plot all metrics including loss, accuracy, precision, recall, and F1 score.

        :param train_loss: List of training loss values.
        :param val_loss: List of validation loss values.
        :param train_acc: List of training accuracy values.
        :param val_acc: List of validation accuracy values.
        :param train_precision: List of training precision values.
        :param val_precision: List of validation precision values.
        :param train_recall: List of training recall values.
        :param val_recall: List of validation recall values.
        :param train_f1: List of training F1 score values.
        :param val_f1: List of validation F1 score values.
        """
        self.plot_loss(train_loss, val_loss)
        self.plot_accuracy(train_acc, val_acc)
        self.plot_precision(train_precision, val_precision)
        self.plot_recall(train_recall, val_recall)
        self.plot_f1(train_f1, val_f1)
