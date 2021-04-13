import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomContrast
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, Layer, BatchNormalization, Lambda, add, concatenate
from CTCLayer import CTCLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential


# This is taken from: https://github.com/kurapan/CRNN/blob/master/models.py
def ctc_lambda_func(args):
    iy_pred, ilabels, iinput_length, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    return tf.reduce_mean(tf.nn.ctc_loss(ilabels, iy_pred, ilabel_length, (iinput_length - 2), logits_time_major=False, blank_index=-1))
    

def buildModel(img_width, img_height, n_output, opt):
    augment = Sequential([
        RandomContrast(0.1),
        RandomRotation(0.1)
    ])
    # Model for prediction
    input_img = Input((img_width, img_height, 1))
    input_img_augmented = augment(input_img)

    # The model is adapted from the CRNN here: https://github.com/kurapan/CRNN/blob/master/models.py
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
    p_8 = MaxPooling2D(pool_size=(2, 2), name='maxpool_8')(bn_7)

    p_8_shape = p_8.get_shape()
    reshape = Reshape(target_shape=(int(p_8_shape[1]), int(p_8_shape[2] * p_8_shape[3])), name='reshape')(p_8)

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

    # The layer simply outputs the loss
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([pred, labels, input_length, label_length])
    # This is extremely strange to read, but it is required because of the way .fit() works.
    # Building a custom training loop (or overriding train_step) would be a much better solution).
    # Basically, this model takes all the inputs and outputs the CTC loss.
    training_model = Model(inputs=[input_img, labels, input_length, label_length], outputs=[ctc_loss], name="end2end_ctc_loss_model")
    if opt:
        # Again, this can be improved... The first parameter here is a sequence of zeros, that we returned in our generator. The second parameter (the output of training_model) is the actual loss.
        training_model.compile(loss={'ctc': lambda _, ctc_loss: ctc_loss}, optimizer=opt)

    return training_model, prediction_model