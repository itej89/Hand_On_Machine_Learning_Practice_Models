import tensorflow as tf
from tensorflow import keras

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        print("Calculating Huber loss. . .")
        error = y_true = y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2 
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}