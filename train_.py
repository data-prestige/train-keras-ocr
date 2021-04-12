import glob
import os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, Layer, BatchNormalization, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from models import CRNN_STN

data_dir = Path("./validation/")
chinese_lp = Path("./chinese_lp/")
validation_lp = Path("./validation/")

_jpg = "*.jpg"

validation_lpimages = sorted(list(map(str, list(validation_lp.glob(_jpg)))))

images = sorted(list(map(str, list(data_dir.glob(_jpg)))))
chines_images = sorted(list(map(str, list(chinese_lp.glob(_jpg)))))
images = images + chines_images

# These can be set as hyper-parameters
img_width = 110
img_height = 470
reduction_factor = 4
# Load inside a TF dataset


validation_dataset = tf.data.Dataset.from_tensor_slices(validation_lpimages)
dataset = tf.data.Dataset.from_tensor_slices(images)


def process_path(image_path):
    # Convert the dataset as:
    # (path) --> (image, label [str], input_len, label_len), 0
    # input_len is always img_width // reduction_factor, should be changed depending on the model.
    # The last 0 is there only for compatibility w.r.t. .fit(). It is ignored afterwards.
    # Load the image and resize
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [img_width, img_height])

    # Get the label and its length
    label = tf.strings.split(image_path, '.jpeg')[0]
    label = tf.strings.split(label, '_')[0]
    label = tf.strings.split(label, ' ')[0]
    label = tf.strings.split(label, '/')[1]
    label = tf.strings.upper(label)
    label_len = tf.strings.length(label)

    return (img, tf.strings.bytes_split(label), img_width // reduction_factor, label_len), tf.zeros(1)


# Apply the preprocessing to each image
dataset = dataset.map(process_path)
validation_dataset = validation_dataset.map(process_path)
# Now we build the dictionary of characters.
# I am assuming every character we have is valid, but this can be changed accordingly.
lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
    num_oov_indices=0, mask_token=None,
)
lookup.adapt(dataset.map(lambda xb, _: xb[1]))  # Note: xb[1] is the label

def convert_string(xb, yb):
    # Simple preprocessing to apply the StringLookup to the label
    return (xb[0], lookup(xb[1]), xb[2], xb[3]), yb

dataset = dataset.map(convert_string)
validation_dataset = validation_dataset.map(convert_string)
## model buiuldi
n_output = len(lookup.get_vocabulary()) + 1 
# Model for prediction
cfg = {"width":img_width, "height":img_height, "nb_channels":1, "ncharacters": n_output}
training_model, _ = CRNN_STN(cfg)

opt = Adam()
# # Again, this can be improved... The first parameter here is a sequence of zeros, that we returned in our generator. The second parameter (the output of training_model) is the actual loss.
training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

# For the training dataset, we apply shuffling and batching. Any data augmentation should go here.
train_dataset = dataset.shuffle(1000).padded_batch(32)
validation_dataset = validation_dataset.shuffle(100).padded_batch(32)
# This was way longer than expected...
#training_model.fit(train_dataset, epochs=10)
# Load inside a TF dataset
# Training
default_callbacks = []

# Add save best model
checkPoint = ModelCheckpoint("cnn_bilstm_ctc_loss_end2end_model.h5", save_weights_only=False, monitor="val_loss", verbose=1, save_best_only=True, mode='min')
default_callbacks = default_callbacks+[checkPoint]
# Add early stopping
earlyStopping=EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, restore_best_weights=True, mode='min') 
default_callbacks = default_callbacks+[earlyStopping]
# Train the deep_model

history = training_model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks=default_callbacks)

