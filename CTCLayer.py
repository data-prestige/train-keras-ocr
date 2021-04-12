from tensorflow import transpose, cast, shape, ones, math, reduce_mean,  nn

from tensorflow import keras
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Layer


class CTCLayer(Layer):

    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name, **kwargs)      
        self.loss_fn = ctc_batch_cost
        
    def call(self, y_true, y_pred):
        batch_len = cast(shape(y_true)[0], dtype="int64")
        input_length = cast(shape(y_pred)[1], dtype="int64")
        
        label_length = cast(shape(math.count_nonzero(y_true, axis=1)), dtype="int64")
        input_length = input_length * ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ones(shape=(batch_len, 1), dtype="int64")

        # loss = nn.ctc_loss(y_true, y_pred, label_length, input_length, logits_time_major=False)
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred