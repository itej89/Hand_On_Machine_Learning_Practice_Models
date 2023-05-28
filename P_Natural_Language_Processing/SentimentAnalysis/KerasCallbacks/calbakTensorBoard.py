class calbakTensorboard:
    @staticmethod
    def getTensorboardCalbak(log_path):

        import os
        from pathlib import Path

        from tensorflow.keras.callbacks import TensorBoard
        
        tensorboard = TensorBoard(log_dir= log_path,
        histogram_freq = 1, embeddings_freq=1, embeddings_metadata={'embedding': 'metadata.tsv'} ,
        write_images=True)

        return tensorboard 