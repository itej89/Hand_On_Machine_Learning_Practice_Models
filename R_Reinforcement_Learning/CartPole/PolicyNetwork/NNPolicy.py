import os, sys
from re import A


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.training.tracking import base

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"




from pathlib import Path


pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent, "KerasCallbacks"))
sys.path.append(os.path.join(pwd.parent, "CustomComponents"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS


import matplotlib.pyplot as plt

# from custom_loss import HuberLoss

class NNPolicy :
    
    def __init__(self, _ObservationCount) -> None:
        self.ObservationCount = _ObservationCount
 
    def create_pipeline(self):
        from sklearn.pipeline import Pipeline
        from Normalize3D import Normalize3D
        
        input_pipeline = Pipeline([
            ('std_Scaler', Normalize3D())
        ])

        return input_pipeline
    
    def runthrough_pipeline(self, full_pipeline, data_frame):
        data_formated = full_pipeline.fit_transform(data_frame)
        return data_formated


    def BuildPolicyNetwork(self):

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=[self.ObservationCount]),
            keras.layers.Dense(5, activation="elu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])


        self.optimizer = keras.optimizers.Adam(lr=0.01)
        self.loss_fn = keras.losses.binary_crossentropy  

        self.PrintModel(model)
        return model


    def get_callbacks(self):
        from calbakTensorBoard import calbakTensorboard
        callbacks = [calbakTensorboard.getTensorboardCalbak()]

        from calbakLearningRateExponentialSchedulerPerEpoch import calbakLearningRateExponentialSchedulerPerEpoch
        lr_decay_callback = calbakLearningRateExponentialSchedulerPerEpoch(model_params, None)
    
        # Modified Checkpoint Callback
        from calbakModelCheckpointEnhanced import calbakModelCheckpointEnhanced
        ckpt_callback = calbakModelCheckpointEnhanced(model_params = model_params, filepath=MODEL_PATH_CONSTANTS.GetCheckPointPath(), monitor='val_loss',
                                    save_best_only=True ,model_params_filepath=MODEL_PATH_CONSTANTS.GetModelParamPath() )

        from tensorflow import keras
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
        lr_perf_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

        callbacks  = callbacks + [lr_perf_scheduler, early_stopping_cb, ckpt_callback]

        return callbacks


    def train_one_step(self, obs, model):
        import numpy as np
        obs = np.array([obs]).reshape((1,4))
        with tf.GradientTape() as tape:
            left_proba = model(obs)
            action = (tf.random.uniform([1,1]) > left_proba)
            y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(self.loss_fn(y_target, left_proba))
            grads = tape.gradient(loss, model.trainable_variables)
            return action, grads
             
    def fit_model(self, Build_Model, train, checkpoint_path=None, model_param_path=None, codings_size = 30, no_epochs=30, batch_size=250, initial_epoch = 0, verbosity=1, get_model = False):
        
       
        from Exp_Sched_Params import Exp_Sched_Params
        model_params = Exp_Sched_Params()

        # Load checkpoint:
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            # Load model:
            from tensorflow.keras.models import load_model
            from tensorflow import keras
            model = load_model(checkpoint_path)

        else:
            model = Build_Model(codings_size)
            MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())       

         

        if get_model:
            return model

        # Start/resume training
        
        # Start/resume training
        model.fit(train,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        validation_data=valid,
        callbacks=self.get_callbacks())



    def PrintModel(self, model):
        print(model.summary())
        # print(f"model  Name : {(model.layers[1].get_weights())}")
        from tensorflow.keras.utils import plot_model
        MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())
        plot_model(model, to_file=MODEL_PATH_CONSTANTS.GetModelVizPath(), 
        show_shapes=True, show_layer_names=True, rankdir="LR")


    def evaluate_model(self, checkpoint_path, test):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)
        print(f"Evaluated Result : {model.evaluate(test, test)}")


    def model_predict(self, checkpoint_path, test_x):
        from tensorflow.keras.models import load_model

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path)
        print(f"Predicted value : {model.predict(test_x)}")
        for x, y in test_x:
            print(f"Actual value : {y}")

        

   
