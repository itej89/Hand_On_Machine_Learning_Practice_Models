class calbakEarlyStopping:
    @staticmethod
    def getEarlyStoppingCalbak(patience=10):

        from tensorflow.keras.callbacks import EarlyStopping

        return EarlyStopping(patience=patience, restore_best_weights=True) 