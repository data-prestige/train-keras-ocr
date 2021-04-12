
import os
import platform
import re
import shutil
import pdb
import glob
import cv2
import timeit
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode 
from tensorflow.io import read_file, decode_png
from tensorflow.image import convert_image_dtype, resize
from tensorflow.strings import unicode_split, reduce_join


"""**Functions Definition**"""

def extract_external_valid_set(external_validation_data_dir, max_length, shuffle=True):
    
    valid_images = np.array(sorted(list(map(str, list(Path(external_validation_data_dir).glob("*.jpg"))))))
    valid_labels = np.array([img.split(os.path.sep)[-1].split(".jpg")[0].split("_")[0].upper() for img in valid_images])
    
    if left_pad == True: 
        valid_labels = np.array([label.strip().rjust(max_length, fill_char) for label in valid_labels])           # left padding
    else: 
        valid_labels = np.array([label.strip().ljust(max_length, fill_char) for label in valid_labels])           # right padding
    
    size = len(valid_images)                # Get the total size of the dataset
    indices = np.arange(size)               # Make an indices array and shuffle it, if required
    if shuffle:
        np.random.shuffle(indices)
    valid_samples = size  # Get the size of training samples    
    
    x_valid, y_valid = valid_images[indices[:valid_samples]], valid_labels[indices[:valid_samples]]         # Split data into training 
    return x_valid, y_valid

def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)                      # Get the total size of the dataset
    indices = np.arange(size)               # Make an indices array and shuffle it, if required
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)  # Get the size of training samples    
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]         # Split data into training 
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]         # and validation sets
    return x_train, x_valid, y_train, y_valid


def plot_predictions(predictions_name, title_plot, dataset, pred_texts, plot_time = 1, plot = False):
    
    for batch in dataset.take(1):
        batch_images = batch["image"]
        
        fig, ax = plt.subplots(4, 4, figsize=(15, 5))
        fig.suptitle(title_plot, fontsize = 16, fontweight = 'bold')
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    plt.savefig("./"+predictions_name)
    if plot == True: 
        plt.show(block=False)
        plt.pause(plot_time)
    plt.close()
    
def save_and_plot_plate(plate_name, plate_img, plot_time = 1, plot = False):
    plt.figure() # generate a new window
    plt.imshow(plate_img)
    plt.axis('off')
            
    plt.savefig(plate_name, dpi = 300, bbox_inches = 'tight', pad_inches = 0)

    if plot == True: 
        plt.show(block=False)
        plt.pause(plot_time)

    plt.close()