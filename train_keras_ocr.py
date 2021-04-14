import glob, os
from pathlib import Path
import tensorflow as tf
import keras_ocr

# Find all the images inside the folder (only the name)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


data_dir = Path("../images/")
validation_lp = Path("../validation/")

# Split into folder and name
_, paths, images = zip(*[p.parts for p in data_dir.glob("*.jpg")])
paths, images = list(paths), list(images)
_, val_paths, val_images = zip(*[p.parts for p in validation_lp.glob("*.jpg")])
val_paths, val_images = list(val_paths), list(val_images)

img_width = 460
img_height = 110
batch_size = 8

# Load inside a TF dataset
dataset = tf.data.Dataset.from_tensor_slices((paths, images))
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_images))
print(f'There are {len(dataset)} training images.')
print(f'There are {len(val_dataset)} validation images.')

def process_path(image_path, image_name):
    # Convert the dataset as:
    # (path, filename) --> (image, label [str])
    # Load the image and resize
    img = tf.io.read_file(".."+ os.sep +image_path + os.sep + image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])

    # Get the label and its length
    label = tf.strings.split(image_name, '_')[0]
    label = tf.strings.upper(label)

    return img, label

# Apply the preprocessing to each image
dataset = dataset.map(process_path)
val_dataset = val_dataset.map(process_path)

# Now we build the dictionary of characters.
# I am assuming every character we have is valid, but this can be changed accordingly.
lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
    num_oov_indices=0, mask_token=None,
)
lookup.adapt(dataset.map(lambda _, yb: tf.strings.bytes_split(yb)))

# Overwrite default width and height.
# Note: if you use the default ones, we might be able to use the pretrained weights.
build_params = keras_ocr.recognition.DEFAULT_BUILD_PARAMS 
build_params['width'] = img_width
build_params['height'] = img_height

recognizer = keras_ocr.recognition.Recognizer(alphabet=lookup.get_vocabulary(), weights=None,
                                              build_params=build_params)

# This is a terrible hack because we are going back to NumPy only to move back to TensorFlow :-(
def train_gen():
  for xb, yb in dataset.repeat():
    yield xb.numpy(), str(yb.numpy(), 'utf-8')

def val_gen():
  for xb, yb in val_dataset.repeat():
    yield xb.numpy(), str(yb.numpy(), 'utf-8')

train_data_gen = recognizer.get_batch_generator(train_gen(), batch_size=batch_size)
val_data_gen = recognizer.get_batch_generator(val_gen(), batch_size=batch_size)        

for xb, yb in train_data_gen:
  print('A')
  break
# xb, yb are basically the same as our code... Maybe we can reuse that part of the code?
# These generators are more or less equivalent to those we build in our notebook.

training_steps = len(dataset) // batch_size
validation_steps = len(val_dataset) // batch_size

recognizer.compile()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('recognizer_borndigital.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_borndigital.csv')
]
recognizer.training_model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=training_steps,
    validation_steps=validation_steps,
    validation_data=val_data_gen,
    callbacks=callbacks,
    epochs=10,
)