from tensorflow.keras.callbacks import LearningRateScheduler


class calbakLearningRatePowerSchedulerPerEpoch(LearningRateScheduler):

    """ Callback class to modify the default learning rate scheduler to operate each batch"""
    def __init__(self, model_params, schedule, verbose=0):
        super(calbakLearningRatePowerSchedulerPerEpoch, self).__init__(self.modify_learningrate, verbose)
        self.model_params = model_params

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def modify_learningrate(self, epoch, lr):
        self.model_params.learning_rate =  (self.model_params.learning_rate) / ((1 + epoch/s)**c)
        return self.model_params.learning_rate
