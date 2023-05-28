class calbakTensorboard:
    @staticmethod
    def getTensorboardCalbak():
        from tensorflow.keras.callbacks import TensorBoard
        
        tensorboard = TensorBoard(log_dir='./logs',
        histogram_freq = 1,
        write_images=True)

        return tensorboard 