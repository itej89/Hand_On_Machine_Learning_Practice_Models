from tensorflow.keras.callbacks import ModelCheckpoint
import pickle


class calbakModelCheckpointEnhanced(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        # Added arguments
        self.model_params_filepath = kwargs.pop('model_params_filepath')
        self.model_params = kwargs.pop('model_params')
        super().__init__(*args, **kwargs)

    # def on_epoch_end(self, epoch, logs=None):
    #     # Run normal flow:
    #     super().on_epoch_end(epoch,logs)
    #      # If a checkpoint was saved, save also the callback
    #     filepath = self.model_params_filepath.format(epoch=epoch + 1, **logs)
    #     if self.epochs_since_last_save == 0 and epoch!=0:
    #         if self.save_best_only:
    #             current = logs.get(self.monitor)
    #             if current == self.best:
    #                 # Note, there might be some cases where the last statement will save on unwanted epochs.
    #                 # However, in the usual case where your monitoring value space is continuous this is not likely
    #                 with open(filepath, "wb") as f:
    #                     pickle.dump(self.model_params, f)
    #         else:
    #             with open(filepath, "wb") as f:
    #                 pickle.dump(self.model_params, f)
    
