# -*- coding: utf-8 -*-
"""Keras-OCR Experiment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TVDgM_xw_QH655ZJe0DQlUxnI33-KHjD
"""

import glob, os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_ocr
from tensorflow.python.ops import bitwise_ops

from vocabolary import LabelConverter
from EditDistanceCallback import EditDistanceCallback

"""## Data loading (tf.data)"""

# Find all the images inside the folder (only the name)


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

data_dir = Path("../images/")
chinese_dir = Path("../chinese_lp/")
validation_lp = Path("../validation/")
label_converter = LabelConverter()

# Split into folder and name
_jpg = "*.jpg"
# Find all the images inside the folder (only the name)
# Split into folder and name
_, paths, images = zip(*[p.parts for p in data_dir.glob(_jpg)])
paths, images = list(paths), list(images)

_, chinese_paths, chinese_images = zip(*[p.parts for p in chinese_dir.glob(_jpg)])
chinese_paths, chinese_images = list(chinese_paths), list(chinese_images)

_, val_paths, val_images = zip(*[p.parts for p in validation_lp.glob(_jpg)])
val_paths, val_images = list(val_paths), list(val_images)

img_width = 200
img_height = 31
reduction_factor = 4
batch_size = 32

# Load inside a TF dataset
chinese_dataset = tf.data.Dataset.from_tensor_slices((chinese_paths, chinese_images))
resia_dataset = tf.data.Dataset.from_tensor_slices((paths, images))
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_images))

print(f'There are {len(chinese_dataset)} training chinese images.')
print(f'There are {len(resia_dataset)} training european images.')
print(f'There are {len(val_dataset)} validation european images.')

augment = tf.keras.Sequential([
         tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
         tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)
    ])


def process_chinese_path(image_path, image_name):
    # Convert the dataset as:
    # (path, filename) --> (image, label [str], input_len, label_len), 0
    # input_len is always img_width // reduction_factor, should be changed depending on the model.
    # The last 0 is there only for compatibility w.r.t. .fit(). It is ignored afterwards.
    # Load the image and resize
    img = tf.io.read_file(".."+ os.sep+image_path + os.sep + image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.dtypes.cast(img, tf.int32)
    img = bitwise_ops.invert(img) # chinese plates bitwise flip
    img = tf.cast(img[:, :, 0], tf.float32) / 255.0
    img = img[:,:, tf.newaxis]
    # Get the label and its length
    label = tf.strings.split(image_name, '.jpg')[0]
    label = tf.strings.split(label, '_')[0]
    label = tf.strings.split(label, ' ')[0]
    label = tf.strings.upper(label)
    label_len = tf.strings.length(label)

    return (img, tf.strings.bytes_split(label), img_width // reduction_factor - 2, label_len), tf.zeros(1)

def process_path(image_path, image_name):
    # Convert the dataset as:
    # (path, filename) --> (image, label [str], input_len, label_len), 0
    # input_len is always img_width // reduction_factor, should be changed depending on the model.
    # The last 0 is there only for compatibility w.r.t. .fit(). It is ignored afterwards.
    # Load the image and resize
    img = tf.io.read_file(".." + os.sep + image_path + os.sep + image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img[:, :, 0], tf.float32) / 255.0
    img = img[:, :, tf.newaxis]
    # Get the label and its length
    label = tf.strings.split(image_name, '.jpg')[0]
    label = tf.strings.split(label, '_')[0]
    label = tf.strings.split(label, ' ')[0]
    label = tf.strings.upper(label)
    label_len = tf.strings.length(label)

    return (img, tf.strings.bytes_split(label), img_width // reduction_factor - 2, label_len), tf.zeros(1)

# Apply the preprocessing to each image
chinese_dataset = chinese_dataset.map(process_chinese_path, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
resia_dataset = resia_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
dataset = chinese_dataset.concatenate(resia_dataset)
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

for (xb, yb, _, yb_len), _ in dataset:
    plt.imshow(xb.numpy()[:, :, 0])
    print(yb)
    print(yb_len)
    break

"""## Build the lookup dictionary"""
# Check the vocabulary
print(label_converter.lookup.get_vocabulary())

def convert_string(xb, yb):
    # Simple preprocessing to apply the StringLookup to the label
    return (xb[0], label_converter.lookup(xb[1]), xb[2], xb[3]), yb

dataset = dataset.map(convert_string)
val_dataset = val_dataset.map(convert_string)

for (xb, yb, xb_len, yb_len), _ in dataset:
    print(yb)
    break

# padded_batch can be used to pad the label appropriately.
# The other values should not require padding.
padded_shapes=(
                ((None, None, None), 
                (img_width // reduction_factor - 2, ), 
                (), 
                ()), (None,))
for (xb, yb, xb_len, yb_len), _ in dataset.padded_batch(batch_size, 
                                            padded_shapes=padded_shapes):
    print(xb.shape)
    print(yb.shape)
    print(xb_len.shape)
    print(yb_len.shape)
    break

"""## Build the model"""

BUILD_PARAMS = keras_ocr.recognition.DEFAULT_BUILD_PARAMS
print(BUILD_PARAMS)

backbone, model, training_model, prediction_model = keras_ocr.recognition.build_model(
            alphabet=keras_ocr.recognition.DEFAULT_ALPHABET, **BUILD_PARAMS)

xb.shape

training_model.summary()

# Load the pretrained weights
weights_dict = keras_ocr.recognition.PRETRAINED_WEIGHTS['kurapan']
model.load_weights(
                    keras_ocr.tools.download_and_verify(url=weights_dict['weights']['top']['url'],
                                              filename=weights_dict['weights']['top']['filename'],
                                              sha256=weights_dict['weights']['top']['sha256']))


training_model.compile(loss=lambda _, y_pred: y_pred, optimizer='rmsprop')

callbacks = [
    EditDistanceCallback(),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, verbose=1, min_lr=0.00001, min_delta=0.1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('recognizer_borndigital.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_borndigital.csv')
]

training_model.fit(
    dataset.shuffle(1000).padded_batch(batch_size, padded_shapes=padded_shapes),
    validation_data=val_dataset.padded_batch(batch_size, padded_shapes=padded_shapes),
    callbacks=callbacks,
    epochs=1000,
)

"""## Decoding"""
y_pred = prediction_model.predict(xb)

alphabet = keras_ocr.recognition.DEFAULT_ALPHABET
blank_label_idx = len(alphabet)

predictions = [''.join([
            alphabet[idx] for idx in y_hat
            if idx not in [blank_label_idx, -1]
        ]) for y_hat in y_pred]

print(predictions)