import glob
import os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, Layer, BatchNormalization, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD

data_dir = Path("../validation/")
chinese_lp = Path("../chinese_lp/")
validation_lp = Path("../validation/")

_jpg = "*.jpg"

images = sorted(list(map(str, list(data_dir.glob(_jpg)))))
images1  = sorted(list(map(str, list(chinese_lp.glob(_jpg)))))
# Split into folder and name
# paths, images = zip(*[p.parts for p in data_dir.glob(_jpg)])
# paths, images = list(paths), list(images)
# paths1, images1 = zip(*[p.parts for p in chinese_lp.glob(_jpg)])
# paths1, images1 = list(paths1), list(images1)
# paths = paths + paths1

# val_paths, val_images = zip(*[p.parts for p in validation_lp.glob(_jpg)])
# val_paths, val_images = list(val_paths), list(val_images)
val_images = sorted(list(map(str, list(validation_lp.glob(_jpg)))))

# These can be set as hyper-parameters
img_width = 460
img_height = 110
reduction_factor = 4
# Load inside a TF dataset

# Load inside a TF dataset
dataset = tf.data.Dataset.from_tensor_slices(images)
val_dataset = tf.data.Dataset.from_tensor_slices(val_images)

print(f'There are {len(dataset)} training images.')
print(f'There are {len(val_dataset)} validation images.')

def process_path(image_path):
    # Convert the dataset as:
    # (path) --> (image, label [str], input_len, label_len), 0
    # input_len is always img_width // reduction_factor, should be changed depending on the model.
    # The last 0 is there only for compatibility w.r.t. .fit(). It is ignored afterwards.
    # Load the image and resize
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.flip_left_right(img)
    img = tf.cast(img[:, :, 0], tf.float32) / 255.0 # Normalization
    img = tf.transpose(img, [1, 0])
    img = img[:, :, tf.newaxis]

    # Get the label and its length
    label = tf.strings.split(image_path, '.jp')[0]
    label = tf.strings.split(label, '_')[0]
    label = tf.strings.split(label, ' ')[0]
    label = tf.strings.split(label, '/')[1]
    label = tf.strings.upper(label)
    label_len = tf.strings.length(label)

    return (img, tf.strings.bytes_split(label), img_width // reduction_factor, label_len), tf.zeros(1)


# Apply the preprocessing to each image
dataset = dataset.map(process_path)
validation_dataset = val_dataset.map(process_path)
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
val_dataset = val_dataset.map(convert_string)


augment = Sequential([
    RandomContrast(0.1),
    RandomRotation(0.1)
])
xb_augm = augment(xb)
n_output = len(lookup.get_vocabulary())
# Model for prediction
input_img = Input((img_width, img_height, 1))
input_img_augmented = augment(input_img)

# The model is adapted from the CRNN here: https://github.com/kurapan/CRNN/blob/master/models.py
c_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')(input_img_augmented)
c_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2')(c_1)
c_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3')(c_2)
bn_3 = BatchNormalization(name='bn_3')(c_3)
p_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(bn_3)

c_4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4')(p_3)
c_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5')(c_4)
bn_5 = BatchNormalization(name='bn_5')(c_5)
p_5 = MaxPooling2D(pool_size=(2, 2), name='maxpool_5')(bn_5)

c_6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_6')(p_5)
c_7 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_7')(c_6)
bn_7 = BatchNormalization(name='bn_7')(c_7)

bn_7_shape = bn_7.get_shape()
reshape = Reshape(target_shape=(int(bn_7_shape[1]), int(bn_7_shape[2] * bn_7_shape[3])), name='reshape')(bn_7)

fc_9 = Dense(128, activation='relu', name='fc_9')(reshape)

lstm_10 = LSTM(128, kernel_initializer="he_normal", return_sequences=True, name='lstm_10')(fc_9)
lstm_10_back = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True, name='lstm_10_back')(fc_9)
lstm_10_add = add([lstm_10, lstm_10_back])

lstm_11 = LSTM(128, kernel_initializer="he_normal", return_sequences=True, name='lstm_11')(lstm_10_add)
lstm_11_back = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True, name='lstm_11_back')(lstm_10_add)
lstm_11_concat = concatenate([lstm_11, lstm_11_back])
do_11 = Dropout(0.25, name='dropout')(lstm_11_concat)

pred = Dense(n_output + 1, kernel_initializer='he_normal', activation=None, name='fc_12')(do_11)
# This is the model to be used for prediction. It only takes the images as input, and outputs
# the corresponding predictions (logits).
prediction_model = Model(inputs=input_img, outputs=pred)

# The model for training will take three more inputs, i.e., the labels, xb_len, and yb_len.
labels = Input(name='labels', shape=[None], dtype='int64')
input_length = Input(name='input_length', shape=[], dtype='int64')
label_length = Input(name='label_length', shape=[], dtype='int64')

# This is taken from: https://github.com/kurapan/CRNN/blob/master/models.py
def ctc_lambda_func(args):
    iy_pred, ilabels, iinput_length, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    # return K.ctc_batch_cost(ilabels, tf.nn.softmax(iy_pred, 2), iinput_length - 2, ilabel_length)
    return tf.reduce_mean(tf.nn.ctc_loss(ilabels, iy_pred, iinput_length - 2, ilabel_length, logits_time_major=False, blank_index=-1))


# The layer simply outputs the loss
ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred, labels, input_length, label_length])

# This is extremely strange to read, but it is required because of the way .fit() works.
# Building a custom training loop (or overriding train_step) would be a much better solution).
# Basically, this model takes all the inputs and outputs the CTC loss.
training_model = Model(inputs=[input_img, labels, input_length, label_length], outputs=[ctc_loss], name="end2end_ctc_loss_model")
# opt = Adam()
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

# Again, this can be improved... The first parameter here is a sequence of zeros, that we returned in our generator. The second parameter (the output of training_model) is the actual loss.
training_model.compile(loss={'ctc': lambda _, ctc_loss: ctc_loss}, optimizer=opt)

# For the training dataset, we apply shuffling and batching. Any data augmentation should go here.
train_dataset = dataset.shuffle(1000).padded_batch(32)
val_dataset = val_dataset.padded_batch(32)

default_callbacks = []

# Add save best model
checkPoint = ModelCheckpoint("cnn_bilstm_ctc_loss_end2end_model.h5", save_weights_only=False, monitor="val_loss", verbose=1, save_best_only=True, mode='min')
default_callbacks = default_callbacks + [checkPoint]

# Add early stopping
lr_reducer = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1, min_lr=0.00001)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, restore_best_weights=True, mode='min') 
default_callbacks = default_callbacks + [earlyStopping]

history = training_model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks=default_callbacks)

