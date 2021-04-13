import glob, os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomContrast
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, Layer, BatchNormalization, Lambda, add, concatenate
from tensorflow.keras import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from build import buildModel, ctc_lambda_func
from vocabolary import LabelConverter
from cv_functions import loadImg


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

data_dir = Path("../images/")
validation_lp = Path("../validation/")
chinese_lp = Path("../chinese_lp/")

label_converter = LabelConverter()

_jpg = "*.jpg"
# Find all the images inside the folder (only the name)
# Split into folder and name
# _, paths, images = zip(*[p.parts for p in data_dir.glob(_jpg)])
# paths, images = list(paths), list(images)
# _, val_paths, val_images = zip(*[p.parts for p in validation_lp.glob(_jpg)])
# val_paths, val_images = list(val_paths), list(val_images)

# Split into folder and name
_, paths, names = zip(*[p.parts for p in data_dir.glob(_jpg)])
paths, names = list(paths), list(names)
images = []
for i, image in enumerate(names):
    images.append(loadImg(paths[i]+"/"+image))

_, chinese_paths, chinese_names = zip(*[p.parts for p in chinese_lp.glob(_jpg)])
chinese_paths, chinese_names = list(chinese_paths), list(chinese_names)
chinese_images = []
for i, image in enumerate(chinese_names):
    chinese_images.append(loadImg(chinese_paths[i]+"/"+image))

paths = paths + list(chinese_paths)
names = names + list(chinese_names)
images = images + chinese_images

_, val_paths, val_names = zip(*[p.parts for p in validation_lp.glob(_jpg)])
val_paths, val_names = list(val_paths), list(val_names)
val_images = []
for i, image in enumerate(val_names):
    val_images.append(loadImg(val_paths[i] + "/" + image))
    

# These can be set as hyper-parameters
img_width = 460
img_height = 110
reduction_factor = 8
# Load inside a TF dataset
# Load inside a TF dataset
dataset = tf.data.Dataset.from_tensor_slices((paths, images))
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_images))

print(f'There are {len(dataset)} training images.')
print(f'There are {len(val_dataset)} validation images.')

# def process_path(image_path, image_name):
#     # Convert the dataset as:
#     # (path) --> (image, label [str], input_len, label_len), 0
#     # input_len is always img_width // reduction_factor, should be changed depending on the model.
#     # The last 0 is there only for compatibility w.r.t. .fit(). It is ignored afterwards.
#     # Load the image and resize
#     img = tf.io.read_file(".."+ os.sep +image_path + os.sep + image_name)
#     img = tf.image.decode_jpeg(img, channels=1)
#     img = tf.image.resize(img, [img_height, img_width])
#     img = tf.image.flip_left_right(img)
#     img = tf.cast(img[:,:, 0], tf.float32) / 255.0  # Normalization
#     img = tf.transpose(img, [1, 0])
#     img = img[:, :, tf.newaxis]
#     # Get the label and its length
#     label = tf.strings.split(image_name, '_')[0]
#     label = tf.strings.upper(label)
#     label_len = tf.strings.length(label)

#     return (img, tf.strings.bytes_split(label), img_width // reduction_factor, label_len), tf.zeros(1)

def process_path(image_name, img):
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.flip_left_right(img)
    img = tf.cast(img[:, :, 0], tf.float32) / 255.0 # Normalization
    #img = tf.image.convert_image_dtype(img, dtype=tf.float32, saturate=True)
    img = tf.transpose(img, [1, 0])
    img = img[:, :, tf.newaxis]

    # Get the label and its length
    label = tf.strings.split(image_name, '_')[0]
    label = tf.strings.upper(label)
    label_len = tf.strings.length(label)

    return (img, tf.strings.bytes_split(label), img_width // reduction_factor, label_len), tf.zeros(1)

# Apply the preprocessing to each image
dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# Now we build the dictionary of characters.
# I am assuming every character we have is valid, but this can be changed accordingly.
dataset = dataset.map(label_converter.convert_string, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(label_converter.convert_string, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# opt = Adam()
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
n_output = label_converter.n_output
training_model, prediction_model = buildModel(img_width, img_height, n_output, opt)

try:
    training_model.load_weights("cnn_bilstm_ctc_loss_end2end_model.h5")
except Exception: print("error loading weights")


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
history = training_model.fit(train_dataset, validation_data=val_dataset, epochs=250, callbacks=default_callbacks)

for (xb, yb, xb_len, yb_len), _ in val_dataset:
    print(yb)
    break
# Create the inverse lookup
y_pred = prediction_model.predict(xb)
y_pred = tf.transpose(y_pred, [1, 0, 2])[2:,:,:]  # Transpose the first dimension and remove the first two time-steps
(y_decoded, _) = tf.nn.ctc_greedy_decoder(y_pred, sequence_length=tf.ones(y_pred.shape[1], dtype=tf.int32)*y_pred.shape[0])
y_decoded_text = label_converter.inv_lookup(y_decoded[0])
results = tf.sparse.to_dense(y_decoded_text)
print(results)
print("allora")
