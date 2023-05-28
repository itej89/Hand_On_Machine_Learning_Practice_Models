import tensorflow as tf
from tensorflow import keras


class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation_name="tanh", **kwargs):
        self.dense = dense
        self.shape = self.dense.input_shape
        self.activation_name = activation_name
        self.activation = keras.activations.get(activation_name)
        super().__init__(**kwargs)
            

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",shape=[self.shape[-1]])
        # super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)
    
    # def compute_output_shape(self, batch_input_shape):
    #     return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])

    def get_config(self):
        base_config =  super().get_config()
        return {**base_config, "dense":self.dense,
        "activation_name":self.activation_name,
        "shape":self.shape}
