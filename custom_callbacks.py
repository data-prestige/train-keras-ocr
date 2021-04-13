import tensorflow as tf
import numpy as np
from IPython.display import display
from PIL import Image
import os
import matplotlib.pyplot as plt



def decode_ctc(args):
    """returns a list of decoded ctc losses"""

    y_pred, input_length = args

    ctc_decoded = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=True)

    return ctc_decoded
    
class PredVisualize(tf.keras.callbacks.Callback):

    def __init__(self, model, val_datagen, lbl_to_char_dict, printing=False):
        """CTC decode the results and visualize output"""
        self.model = model
        self.iterable = val_datagen
        self.iter_obj = iter(self.iterable)
        self.printing = printing
        self.lbl_to_char_dict = lbl_to_char_dict

    def get_validation_batch(self):
        while True:
            try:
                next_obj = next(self.iter_obj)
            except StopIteration:
                self.iter_obj = iter(self.iterable)
                next_obj = next(self.iter_obj)
            return next_obj

    def on_epoch_end(self, batch, logs=None):
        #make a batch of data
        batch_imgs, batch_labels = self.get_validation_batch()
        #predict from batch
        y_preds = tf.nn.softmax(self.model.predict(batch_imgs), axis=2)
        batch_len = tf.cast(tf.shape(batch_labels)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_preds)[1], dtype="int64")
        #reshape for the loss, add that extra meaningless dimension
        label_length = tf.math.count_nonzero(batch_labels, axis=1)
        input_length = input_length * tf.ones(shape=(batch_len), dtype="int64")
        #call the ctc decode
        pred_tensor, _ = decode_ctc([y_preds, input_length])
        pred_labels = tf.keras.backend.get_value(pred_tensor[0]) + 1

        #map back to strings
        predictions = ["".join([self.lbl_to_char_dict[i] for i in word if i!=0]) for word in pred_labels.tolist()]
        truths = ["".join([self.lbl_to_char_dict[i] for i in word if i!=0]) for word in batch_labels.tolist()]

        # combine the images and print at screen if printing is on
        # transpose first back to original form
        if self.printing:
            imgs_list_arr_T = [img.transpose((1, 0, 2)) for img in batch_imgs]
            imgs_comb = np.hstack(imgs_list_arr_T) * 255
            imgs_comb = Image.fromarray(imgs_comb.astype(np.uint8), 'RGB')
            # display(imgs_comb.resize((40 * imgs_comb.width, 40 * imgs_comb.height), Image.NEAREST))
            plt.figure(figsize=(40, 40))
            plt.imshow(imgs_comb)
            plt.show()

        print('predictions {}'.format(predictions))

def make_save_model_cb(folder="saved_models"):
    """save model weights after each epoch callback
        Should really save the whole model but for some reason doesn't work in tf2"""

    filename = "weights.h5"
    filepath = os.path.join(os.getcwd(), folder, filename)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, save_best_only=False)

    return callback

