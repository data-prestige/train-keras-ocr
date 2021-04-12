
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

from preprocessing import Preprocessing
import tensorflow as tf
from tensorflow.config.experimental import list_physical_devices
from tensorflow.python.client import device_lib

from tensorflow.strings import unicode_split, reduce_join
from tensorflow import transpose, cast, shape, ones
from tensorflow.io import read_file, decode_png
from tensorflow.image import convert_image_dtype, resize
from tensorflow.data import Dataset

from tensorflow import keras
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam
#from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, Layer
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from functions import split_data
from build import build_model

"""**Globals**"""

start_global_time = timeit.default_timer()

# Operating System
OS = platform.system()                    # returns 'Windows', 'Linux', etc

physical_devices= list_physical_devices('GPU')
print("Number of GPUs", len(physical_devices))
print("\nCPUs and GPUs details: \n")
print(device_lib.list_local_devices())


early_stopping = False
early_stopping_patience = 10

save_best_model = True

best_model_weights_filename = "./cnn_bilstm_ctc_loss_end2end_model.h5"
best_model_architecture_filename = "./cnn_bilstm_ctc_loss_end2end_model.json"

left_pad = True
fill_char = '0'
fixed_max_length = True
labels_max_length = 12
# Desired image dimensions
img_width = 200
img_height = 50
epochs = 20
batch_size = 128
data_augmentation = True
# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4
# View
plot = False

data_dir = Path("./images/")
chinese_lp = Path("./chinese_lp/")
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
chines_images = sorted(list(map(str, list(chinese_lp.glob("*.jpg")))))
labelsch = [img.split(os.path.sep)[-1].split(".jpg")[0].split("_")[0].strip().upper() for img in chines_images]
labels = [img.split(os.path.sep)[-1].split(".jpg")[0].split("_")[0].split(" ")[0].strip().upper() for img in images]
labels = labels + labelsch
images = images + chines_images
max_length = max([len(label) for label in labels])
characters = sorted(list(set(char for label in labels for char in label)))

def padLabel(label):
    return label + "*" * (max_length-len(label))
labels = [padLabel(label) for label in labels]

n_output = len(characters) + 2
# characters.insert(0, "*")

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)
#print(labels)
"""**Preprocessing**"""
# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]   # Use greedy search. For complex tasks, you can use beam search
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
    
def plot_dataset(dataset_name, title_plot, dataset, plot_time = 1, plot = False):
    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
    fig.suptitle(title_plot, fontsize = 16, fontweight = 'bold')
    for batch in dataset.take(1):
        images = batch["image"]
        labels = batch["label"]
        for i in range(16):
            img = (images[i] * 255).numpy().astype("uint8")
            label = reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
            ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")
    plt.savefig("./"+dataset_name)
    if plot == True: 
        plt.show(block=False)
        plt.pause(plot_time)
    plt.close()


def encode_single_sample(img_path, label):
    img = read_file(img_path)                                                           # Read image
    img = decode_png(img, channels=1)                                                   # Decode and convert to grayscale
    img = convert_image_dtype(img, tf.float32)                                          # Convert to float32 in [0, 1] range
    img = resize(img, [img_height, img_width])                                          # Resize to the desired size
    img = transpose(img, perm=[1, 0, 2])
    label = char_to_num(unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}
    
if data_augmentation == True: 
    image_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, fill_mode="nearest")

char_to_num = StringLookup(vocabulary=list(characters), num_oov_indices=0, mask_token="*", oov_token="*")          # Mapping characters to integers
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token="*", oov_token="*", invert=True)    # Mapping integers back to original characters

x_train, x_valid_intern, y_train, y_valid_intern = split_data(np.array(images), np.array(labels))

"""## Create `Dataset` objects"""
train_dataset = Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

plot_dataset("training_set.jpg", "Sample of Training Set", train_dataset, 2, plot = plot)

internal_validation_dataset = Dataset.from_tensor_slices((x_valid_intern, y_valid_intern))
internal_validation_dataset = (internal_validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
 
"""**Training**"""
# Get the deep_model
deep_model = build_model(n_output, img_width, img_height)
deep_model.summary()

# Training
default_callbacks = []

# Add save best model
checkPoint = ModelCheckpoint(best_model_weights_filename, save_weights_only=False, monitor = "val_loss", verbose=1, save_best_only=True, mode='min')
default_callbacks = default_callbacks+[checkPoint]
# Add early stopping
earlyStopping=EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=early_stopping_patience, verbose=0, restore_best_weights = True, mode='min') 
default_callbacks = default_callbacks+[earlyStopping]
# Train the deep_model

history = deep_model.fit(train_dataset, validation_data = internal_validation_dataset, epochs = epochs, callbacks = default_callbacks)

