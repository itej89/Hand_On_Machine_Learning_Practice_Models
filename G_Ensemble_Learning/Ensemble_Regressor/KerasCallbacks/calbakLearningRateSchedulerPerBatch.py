from tensorflow.keras.callbacks import LearningRateScheduler


class calbakLearningRateSchedulerPerBatch(LearningRateScheduler):

    """ Callback class to modify the default learning rate scheduler to operate each batch"""
    def __init__(self, model_params, schedule, verbose=0):
        super(calbakLearningRateSchedulerPerBatch, self).__init__(self.modify_learningrate, verbose)
        self.model_params = model_params

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(calbakLearningRateSchedulerPerBatch, self).on_epoch_begin(self.model_params.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(calbakLearningRateSchedulerPerBatch, self).on_epoch_end(self.model_params.count, logs)
        self.model_params.count += 1

    def modify_learningrate(self, epoch, lr):
        self.model_params.learning_rate =  ((self.model_params.learning_rate - self.model_params.min_learning_rate) * self.model_params.decay_rate ** self.model_params.count + self.model_params.min_learning_rate)
        return self.model_params.learning_rate