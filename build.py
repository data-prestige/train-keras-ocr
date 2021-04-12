from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, Layer
from CTCLayer import CTCLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(n_output, img_width, img_height):

    # Input Stage
    input_img = Input(shape=(img_width, img_height, 1), name="image", dtype="float32")                                  # Inputs to the model
    labels = Input(name="label", shape=(None,), dtype="float32")
    
    # Convolutionary Feature Extractor 
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)  # First conv block
    x = MaxPooling2D((2, 2), name="pool1")(x)
    x = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)                                                                           # Second conv block
    

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = Reshape(target_shape=new_shape, name="reshape")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    x = Dropout(0.2)(x)
    
    # Encoder Layer with BiLSTM 
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)
    
    # Output layer
    x = Dense(n_output, activation="softmax", name="dense2")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)                           # Add CTC layer for calculating CTC loss at each step
    
    # Define the model
    deep_model = Model(inputs=[input_img, labels], outputs=output, name="end2end_ctc_loss_model")
    
    opt = Adam()                            # Optimizer

    deep_model.compile(optimizer=opt)            # Compile the deep_model and return

    return deep_model