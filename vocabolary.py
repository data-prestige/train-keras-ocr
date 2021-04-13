import tensorflow as tf# Now we build the dictionary of characters.
# I am assuming every character we have is valid, but this can be changed accordingly.


class LabelConverter:

    def __init__(self):
        alphabet = list("0123456789ABCDEFGHJKILMNOPQRSTUVWXYZ")
        self.lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=alphabet, num_oov_indices=0, mask_token=None,)
        self.inv_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.lookup.get_vocabulary(), invert=True, mask_token=None)
        self.n_output = len(self.lookup.get_vocabulary())

    def convert_string(self, xb, yb):
        # Simple preprocessing to apply the StringLookup to the label
        return (xb[0], self.lookup(xb[1]), xb[2], xb[3]), yb



