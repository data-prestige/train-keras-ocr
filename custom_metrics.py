import tensorflow as tf
import numpy as np
from IPython.display import display
from PIL import Image
import os
import matplotlib.pyplot as plt



# class BinaryTruePositives(tf.keras.metrics.Metric):

#   def __init__(self, name='binary_true_positives', **kwargs):
#     super(BinaryTruePositives, self).__init__(name=name, **kwargs)
#     self.true_positives = self.add_weight(name='tp', initializer='zeros')

#   def update_state(self, y_true, y_pred, sample_weight=None):
#     y_true = tf.cast(y_true, tf.bool)
#     y_pred = tf.cast(y_pred, tf.bool)

#     values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
#     values = tf.cast(values, self.dtype)
    
#     if sample_weight is not None:
#       sample_weight = tf.cast(sample_weight, self.dtype)
#       sample_weight = tf.broadcast_to(sample_weight, values.shape)
#       values = tf.multiply(values, sample_weight)
#     self.true_positives.assign_add(tf.reduce_sum(values))

#   def result(self):
#     return self.true_positives


class EditDistance(tf.keras.metrics.Metric):

    def __init__(self, name='edit_distance', **kwargs):
        super(EditDistancePositives, self).__init__(name=name, **kwargs)
        self.edit_distance = self.add_weight(name='ed', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        indices, values, dense_shape = y_true
        labels_pred_pl = tf.SparseTensor(indices, values, dense_shape)
        indices, values, dense_shape = y_pred
        labels_true_pl = tf.SparseTensor(indices, values, dense_shape)
        edit_op = tf.edit_distance(labels_true_pl, labels_pred_pl, normalize=True)
        self.edit_distance.assign_add(tf.reduce_sum(edit_op))

    def result(self):
        return self.edit_distance

