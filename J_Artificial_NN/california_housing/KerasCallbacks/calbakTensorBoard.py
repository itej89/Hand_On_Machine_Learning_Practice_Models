class calbakTensorboard:
    @staticmethod
    def getTensorboardCalbak(log_path):

        import os
        from pathlib import Path

        from tensorflow.keras.callbacks import TensorBoard
        
        tensorboard = TensorBoard(log_dir= log_path,
        histogram_freq = 1,
        write_images=True)

        return tensorboard 