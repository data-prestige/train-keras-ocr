import tensorflow as tf
import numpy as np
from IPython.display import display
from PIL import Image
import os
import matplotlib.pyplot as plt
from Levenshtein import distance
import keras_ocr

one32 = -np.ones(1, dtype=np.int32)[0]
one64 = -np.ones(1, dtype=np.int64)[0]
img_width = 200
img_height = 31
reduction_factor = 4
batch_size = 32
alphabet = keras_ocr.recognition.DEFAULT_ALPHABET
blank_label_idx = len(alphabet)
padded_shapes = (((None, None, None), 
                (img_width // reduction_factor - 2, ), 
                (), 
                ()), (None,))
padding_values = ((-1.0, one64, one32, one32), -1.0)

def decode_predictions(y_pred):
  return [''.join([
            alphabet[idx] for idx in y_hat
            if idx not in [blank_label_idx, -1]
        ]) for y_hat in y_pred]

# Define a custom callback for evaluating the edit distance (terribly slow and inefficient).
class EditDistanceCallback(tf.keras.callbacks.Callback):

    def __init__(self, prediction_model, val_dataset, batch_size, patience=0):
        super(EditDistanceCallback, self).__init__()
        self.prediction_model = prediction_model
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.edit_distances = []
      
    def on_epoch_begin(self, epoch, logs=None):
        edit_distance = 0
        for (xb, yb, xb_len, yb_len), _ in self.val_dataset.padded_batch(batch_size,
          padded_shapes=padded_shapes, padding_values=padding_values):
          ypred = self.prediction_model(xb)
          # Decode the predictions (list(int) --> string)
          ypred_decoded = decode_predictions(ypred)
          yb_decoded = decode_predictions(yb)
          # For each pair, compute the distance
          edit_distance += sum(distance(y1, y2) for y1, y2 in zip(ypred_decoded, yb_decoded))
        self.edit_distances.append(edit_distance / len(self.val_dataset))
        print(f'Edit distance at epoch {epoch+1} is {self.edit_distances[-1]}.')


