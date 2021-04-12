import os
import argparse
import string
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json, load_model
from spatial_transformer import SpatialTransformer


for (xb, yb, xb_len, yb_len), _ in val_dataset:
    print(yb)
    break
# Create the inverse lookup
inv_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=lookup.get_vocabulary(), invert=True, mask_token=None)
y_pred = prediction_model.predict(xb)
y_pred = tf.transpose(y_pred, [1, 0, 2])[2:,:,:]  # Transpose the first dimension and remove the first two time-steps
(y_decoded, _) = tf.nn.ctc_greedy_decoder(y_pred, sequence_length=tf.ones(y_pred.shape[1], dtype=tf.int32)*y_pred.shape[0])
y_decoded_text = inv_lookup(y_decoded[0])
results = tf.sparse.to_dense(y_decoded_text)
print(results)