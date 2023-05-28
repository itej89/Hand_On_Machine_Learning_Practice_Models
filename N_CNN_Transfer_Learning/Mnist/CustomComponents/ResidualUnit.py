import re
import tensorflow as tf
from tensorflow import keras

from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters=64, strides=1, activation_name="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.activation_name = activation_name
       

    def build(self, batch_input_shape):
        self.activation = keras.activations.get(self.activation_name)
        self.main_layers = [
            DefaultConv2D(self.filters, strides=self.strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(self.filters),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if self.strides > 1:
            self.skip_layers = [
                DefaultConv2D(self.filters, kernel_size=1, strides=self.strides),
                keras.layers.BatchNormalization()
            ]
        super().build(batch_input_shape)

    def call(self, X):
        Z = X
        for layer in self.main_layers:
            Z = layer(Z)
        
        skip_Z = X
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        
        return self.activation(Z+skip_Z)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])

    def get_config(self):
        base_config =  super().get_config()
        return {**base_config, "filters":self.filters,
        "strides":self.strides,
        "activation_name":self.activation_name}
