import tensorflow as tf
from tensorflow import keras


class LNSimpleRNN(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units
        self.activation = activation
            

    def build(self, batch_input_shape):
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(self.units,activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(self.activation)
        super().build(batch_input_shape)

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])

    def get_config(self):
        base_config =  super().get_config()
        return {**base_config, "state_size":self.state_size,
        "output_size":self.output_size,
        "activation":self.activation,
        "units":self.units}
