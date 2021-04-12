
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
from tensorflow.data import Dataset
from tensorflow.strings import unicode_split, reduce_join
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from functions import split_data
from tensorflow.image import convert_image_dtype, resize
from tensorflow import transpose, cast, shape, ones

"""**Inference**"""
import timeit
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json, load_model
from CTCLayer import CTCLayer
# from train import decode_batch_predictions, reduce_join, encode_single_sample, num_to_char, split_data
from functions import plot_predictions
from tensorflow.io import read_file, decode_png
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode


img_width = 200
img_height = 50
batch_size = 16

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
    label = char_to_num(unicode_split(label, input_encoding="UTF-8"))                   # Map the characters in label to numbers
    return {"image": img, "label": label}

best_model_weights_filename = "./cnn_bilstm_ctc_loss_end2end_model.h5"

data_dir = Path("./validation/")
# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0].split("_")[0].split(" ")[0].strip().upper() for img in images]
max_length = max([len(label) for label in labels])
characters = sorted(list(set(char for label in labels for char in label)))

def padLabel(label):
    return label + "*" * (max_length-len(label))
labels = [padLabel(label) for label in labels]

n_output = len(characters) + 1
# characters.insert(0, "*")


char_to_num = StringLookup(vocabulary=list(characters), num_oov_indices=0, mask_token="*", oov_token="*")          # Mapping characters to integers
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token="*", oov_token="*", invert=True)    # Mapping integers back to original characters


x_train, x_valid_intern, y_train, y_valid_intern = split_data(np.array(images), np.array(labels))

internal_validation_dataset = Dataset.from_tensor_slices((x_valid_intern, y_valid_intern))
internal_validation_dataset = (internal_validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
 
# Inference
deep_model = load_model(best_model_weights_filename, custom_objects = {'CTCLayer': CTCLayer})
  
# Get the prediction deep_model by extracting layers till the output layer
prediction_model = Model(deep_model.get_layer(name="image").input, deep_model.get_layer(name="dense2").output)    # dense2 becomes the output

start_infer_time = timeit.default_timer()
#  Let's check results on some validation samples
for batch in internal_validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)
    
print(orig_texts)
print(pred_texts)

internal_matchs = len([w for w in pred_texts if w in orig_texts])
internal_acc = internal_matchs / len(orig_texts)
internal_title1 = "\nNumber of matches between predictions and ground truth on Internal Validation Set: "+str(internal_matchs)+"\n"
internal_title2 = "Accuracy on Internal Validation Set: " +  str(int(internal_acc * 100)) + "%\n\n"
print(internal_title1)
print(internal_title2)

plot_predictions("predictions_internal.jpg", internal_title2, internal_validation_dataset, pred_texts, 2, plot = False)

end_infer_time = timeit.default_timer()
print("Inference Time: ", (end_infer_time - start_infer_time) / 60.0)
