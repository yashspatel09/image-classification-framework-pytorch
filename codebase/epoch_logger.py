"""
EventLogger Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to log events and metrics into a CSV file.
"""

import csv
import os

class EventLogger:
    def __init__(self, file_name, fields):
        """
        Initialize the EventLogger with a file name and field names.

        :param file_name: Name of the CSV file where logs will be stored.
        :param fields: List of field names (column names) for the CSV file.
        """
        self.file_name = file_name
        self.fields = fields
        self.init_file()  # Initialize the file and write header if it doesn't exist

    def init_file(self):
        """
        Initialize the CSV file. If the file doesn't exist, create it and write the header.
        """
        # Check if the file already exists
        if not os.path.isfile(self.file_name):
            # Open the file in write mode and create a CSV DictWriter
            with open(self.file_name, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()  # Write the header to the CSV file

    def log_event(self, epoch, metrics, phase='', loss=0):
        """
        Log an event with epoch, metrics, phase, and loss information.

        :param epoch: Current epoch number.
        :param metrics: Dictionary containing metrics to be logged.
        :param phase: Phase of the event (e.g., 'train', 'validation').
        :param loss: Loss value for the current epoch.
        """
        # Open the file in append mode to add new rows
        with open(self.file_name, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            # Prepare the row with epoch, loss, and phase information
            row = {'epoch': epoch, 'loss': loss, 'phase': phase}
            row.update(metrics)  # Update the row with the metrics
            writer.writerow(row)  # Write the row to the CSV file
