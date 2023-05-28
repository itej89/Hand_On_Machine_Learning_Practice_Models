import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.python.keras.layers.serialization import serialize
from tensorflow.python.ops.gen_parsing_ops import decode_raw

K = keras.backend

class Sampling(keras.layers.Layer):

    def call(self, X):
        mean, log_var = X
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    


