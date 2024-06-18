"""
Image Labeling Module
Author: Yash Patel
Copyright: 2024
Description: This module provides functionality to add text labels to images and save them.
"""

from PIL import Image, ImageDraw, ImageFont
import os
from config import Config

# Initialize configuration instance
cfg = Config()

def save_image_with_label(image_path, label, save_path, text_color=(255, 0, 0), font_size=20):
    """
    Add a text label to an image and save it.

    :param image_path: Path to the input image.
    :param label: Text label to add to the image.
    :param save_path: Path to save the edited image.
    :param text_color: Color of the text label.
    :param font_size: Size of the text label.
    """
    # Open an image file
    with Image.open(image_path) as img:
        # Initialize the drawing context with the image object as background
        draw = ImageDraw.Draw(img)
        # Load a font
        font = ImageFont.truetype("arial.ttf", font_size)
        # Position the text at the bottom-right
        text_position = (img.width - font_size, img.height - font_size)
        # Draw the text on the image
        draw.text(text_position, label, fill=text_color, font=font)
        # Save the edited image
        img.save(save_path)

def save_comparison_image(input_path, true_label, pred_label, save_path, text_color=(255, 0, 0), font_size=20):
    """
    Add true and predicted labels to an image and save it.

    :param input_path: Path to the input image.
    :param true_label: True label text.
    :param pred_label: Predicted label text.
    :param save_path: Path to save the edited image.
    :param text_color: Color of the text labels.
    :param font_size: Size of the text labels.
    """
    # Open an image file
    with Image.open(input_path) as img:
        # Initialize the drawing context with the image object as background
        draw = ImageDraw.Draw(img)
        # Load a font
        font = ImageFont.truetype("arial.ttf", font_size)
        # Position the text at the top-left and top-right
        true_label_position = (0, 0)
        pred_label_position = (img.width - 2 * font_size, 0)
        # Draw the true label on the image
        draw.text(true_label_position, 'True: ' + true_label, fill=text_color, font=font)
        # Draw the predicted label on the image
        draw.text(pred_label_position, 'Pred: ' + pred_label, fill=text_color, font=font)
        # Save the edited image
        img.save(cfg.plotname + save_path)

# Example usage:
# save_image_with_label('path_to_image.png', 'label', 'path_to_save_image.png')
# save_comparison_image('path_to_input_image.png', 'true_label', 'predicted_label', 'path_to_save_image.png')
