import glob, os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from vocabolary import LabelConverter

import keras_ocr
label_converter = LabelConverter()

# Find all the images inside the folder (only the name)
validation_lp = Path("./test/")

# Split into folder and name
val_paths, val_images = zip(*[p.parts for p in validation_lp.glob("*.jpg")])
val_paths, val_images = list(val_paths), list(val_images)

img_width = 200
img_height = 31
batch_size = 8

# Load inside a TF dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_images))

print(f'There are {len(val_dataset)} validation images.')

def process_path(image_path, image_name):
    # Convert the dataset as:
    # (path, filename) --> (image, label [str])
    # Load the image and resize
    img = tf.io.read_file("."+ os.sep +image_path + os.sep + image_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # img = tf.cast(img[:, :, :], tf.int8)
    # Get the label and its length
    label = tf.strings.split(image_name, '.jpg')[0]
    label = tf.strings.split(label, '_')[0]
    label = tf.strings.split(label, ' ')[0]
    label = tf.strings.upper(label)

    return img, label

# Apply the preprocessing to each image
val_dataset = val_dataset.map(process_path)

"""## Build and train the keras-ocr recognizer"""
build_params = keras_ocr.recognition.DEFAULT_BUILD_PARAMS 
build_params['width'] = img_width
build_params['height'] = img_height
# Version with custom vocabulary
recognizer = keras_ocr.recognition.Recognizer(alphabet=label_converter.lookup.get_vocabulary(), weights=None, build_params=build_params)
recognizer.prediction_model.load_weights("recognizer_borndigital.h5")
# print(recognizer.alphabet)

def val_gen():
  for xb, yb in val_dataset.repeat():
    yield xb.numpy(), str(yb.numpy(), 'utf-8')

# For models 1-2, remove lowercase
val_data_gen = recognizer.get_batch_generator(val_gen(), batch_size=batch_size, lowercase=True)

# xb, yb are basically the same as our code... Maybe we can reuse that part of the code?
# These generators are more or less equivalent to those we build in our notebook.
validation_steps = len(val_dataset) // batch_size

# print(validation_steps)

recognizer.compile()
predictions = []
labels = []
correct = 0
for xb, yb in val_dataset:
  # plt.imshow(xb.numpy())
  prediction = recognizer.recognize(xb.numpy())
  predictions.append(prediction)
  label = yb.numpy().decode("utf-8")
  labels.append(label)
  eq = prediction == label
  if eq:
    correct += 1
  else: print(prediction, label, eq)

# external_match = len([w for w in predictions if w in labels])
acc = correct / len(labels) * 100

print("targhe correttamente lette:{}, accuratezza: {}".format(correct, acc))
# recognizer.recognize(xb.numpy())