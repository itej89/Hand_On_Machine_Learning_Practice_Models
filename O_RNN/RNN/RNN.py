from operator import add
from tensorflow.keras import activations, layers
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.ops.gen_math_ops import mod
import os, sys

import tensorflow as tf
from tensorflow import keras
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
sys.path.append(os.path.join(pwd.parent, "PipelineTransformers"))
sys.path.append(os.path.join(pwd.parent, "CustomComponents"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS_CSV, LABELS_PKL



class rnn_test :


    def get_random_time_series_datasets(self, _instances, _feature_len, _target_len):
        from LoadData import LoadData
        objLoadData = LoadData()
        instances = _instances
        feature_len = _feature_len
        target_len = _target_len

        series = objLoadData.load_generated_data(instances, feature_len , target_len)
         
        import numpy as np
        Y = np.empty((instances, feature_len, target_len))
        for step_ahead in range(1, target_len+1):
            Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead+feature_len, 0]

        X_train, y_train = series[:7000, :feature_len], Y[:7000]
        X_valid, y_valid = series[7000:9000, :feature_len], Y[7000:9000]
        X_test, y_test = series[9000:, :feature_len], Y[9000:]
       
        return X_train, y_train, X_valid, y_valid, X_test, y_test



    def get_random_datasets(self, _instances, _feature_len, _target_len):
        from LoadData import LoadData
        objLoadData = LoadData()
        instances = _instances
        feature_len = _feature_len
        target_len = _target_len
         
        series = objLoadData.load_generated_data(instances, feature_len, target_len)
        X_train, y_train = series[:7000, :feature_len], series[:7000, -1*target_len:]
        X_valid, y_valid = series[7000:9000, :feature_len], series[7000:9000, -1*target_len:]
        X_test, y_test = series[9000:, :feature_len], series[9000:, -1*target_len:]
       
        return X_train, y_train, X_valid, y_valid, X_test, y_test



#region Sequence to vector Method 1
    def Build_Keras_rnn_Model(self):

        model = keras.models.Sequential([
            keras.layers.SimpleRNN(20, return_sequences=True ,input_shape=[None, 1]),
            keras.layers.SimpleRNN(20, return_sequences=False),
            keras.layers.Dense(1)
        ])


        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model

    
    def predict_next_series(self,  checkpoint_path, test_x, series_len):
        from tensorflow.keras.models import load_model
        import numpy as np

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path, custom_objects={"lr": self.get_lr_metric(optimizer)})

        for step_ahead in range(series_len):
            y_pred_one = model.predict(test_x[:, step_ahead:])[:, np.newaxis, :]
            test_x = np.concatenate([test_x, y_pred_one], axis=1)
            print(' '.join(map(str, test_x))) 
        print(f"Predicted value : {test_x[:, -1*series_len:]}")
#endregion


    def Build_Keras_rnn_multi_out_Model(self):

        model = keras.models.Sequential([
            keras.layers.SimpleRNN(20, return_sequences=True ,input_shape=[None, 1]),
            keras.layers.SimpleRNN(20, return_sequences=False),
            keras.layers.Dense(10)
        ])


        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model


    def Build_Keras_rnn_Seq_to_Seq_Model(self):


        model = keras.models.Sequential([
            keras.layers.SimpleRNN(20, return_sequences=True ,input_shape=[None, 1]),
            keras.layers.SimpleRNN(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])


        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model

    def Build_Keras_rnn_Seq_to_Seq_Layer_Normalized_Model(self):

        from LNSimpleRNN import LNSimpleRNN
        model = keras.models.Sequential([
            keras.layers.RNN(LNSimpleRNN(20), return_sequences=True ,input_shape=[None, 1]),
            keras.layers.RNN(LNSimpleRNN(20), return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model

    def Build_Keras_rnn_Seq_to_Seq_LSTM_Model(self):

        model = keras.models.Sequential([
            keras.layers.LSTM(20, return_sequences=True ,input_shape=[None, 1]),
            keras.layers.LSTM(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model


    def auto_predict_series(self,  checkpoint_path, test_x, series_len):
        from tensorflow.keras.models import load_model
        import numpy as np

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path, custom_objects={"lr": self.get_lr_metric(optimizer)})

        for step_ahead in range(series_len):
            y_pred_one = model.predict(test_x[:, step_ahead:])[:, np.newaxis, :]
            # print(f"pred : {y_pred_one[:, -1, -1][:,:,np.newaxis]}")
            # print(f"test_x : {test_x}")
            test_x = np.concatenate([test_x, y_pred_one[:, -1, -1][:,:,np.newaxis]], axis=1)
            # print(' '.join(map(str, test_x))) 
        print(f"Predicted value : {np.squeeze(test_x)}")


    def Build_Keras_rnn_Seq_to_Seq_1D_Conv_Model(self):

        model = keras.models.Sequential([
            keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid"),
            keras.layers.GRU(20, return_sequences=True ,input_shape=[None, 1]),
            keras.layers.GRU(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ])

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model

    def auto_predict_1D_Conv_series(self,  checkpoint_path, test_x, series_len):
        from tensorflow.keras.models import load_model
        import numpy as np

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path, custom_objects={"lr": self.get_lr_metric(optimizer)})

        for step_ahead in range(series_len):
            y_pred_one = model.predict(test_x[:, step_ahead:])[:, np.newaxis, :]
            # print(f"pred : {y_pred_one[:, -1, -1][:,:,np.newaxis]}")
            print(f"pred shape : {y_pred_one[:, -1, -1][:,:,np.newaxis].shape}")
            print(f"test_x shape : {test_x[:, step_ahead:].shape}")
            test_x = np.concatenate([test_x, y_pred_one[:, -1, -1][:,:,np.newaxis]], axis=1)
            # print(' '.join(map(str, test_x))) 
        print(f"Predicted value : {np.squeeze(test_x)}")

    def Build_Keras_WAVNET_Model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None, 1]))
        for rate in (1, 2, 3, 4)  * 2:
            model.add(keras.layers.Conv1D(filters=20, kernel_size=2, 
            padding="causal", activation="relu", dilation_rate=rate))

        model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
        

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="mean_squared_error",
        optimizer=optimizer, metrics=["MeanSquaredError", lr_metric])

        return model

    def auto_predict_WavNet_series(self,  checkpoint_path, test_x, series_len):
        from tensorflow.keras.models import load_model
        import numpy as np

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path, custom_objects={"lr": self.get_lr_metric(optimizer)})

        for step_ahead in range(series_len):
            y_pred_one = model.predict(test_x[:, step_ahead:])[:, np.newaxis, :]
            # print(f"pred : {y_pred_one[:, -1, -1][:,:,np.newaxis]}")
            # print(f"pred shape : {y_pred_one[:, -1, -1][:,:,np.newaxis].shape}")
            # print(f"test_x shape : {test_x[:, step_ahead:].shape}")
            test_x = np.concatenate([test_x, y_pred_one[:, -1, -1][:,:,np.newaxis]], axis=1)
            # print(' '.join(map(str, test_x))) 
        print(f"Predicted value : {np.squeeze(test_x)}")



    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


    def fit_model(self, mode_builder, train_x, train_y, valid_x, valid_y, checkpoint_path=None, model_param_path=None, no_epochs=30, initial_epoch = 0, verbosity=1, get_model = False):
        
        from Model_Params import Model_Params
        model_params = Model_Params() 

        from calbakTensorBoard import calbakTensorboard
        from calbakEarlyStopping import calbakEarlyStopping
        callbacks = [calbakTensorboard.getTensorboardCalbak(MODEL_PATH_CONSTANTS().GetLogsPath()),
        calbakEarlyStopping.getEarlyStoppingCalbak()]


        # Load checkpoint:
        if checkpoint_path is not None:
            # Load model:
            from tensorflow.keras.models import load_model
            from ResidualUnit import ResidualUnit
            from ResidualUnit import DefaultConv2D
            model = load_model(checkpoint_path)
            for layer in model.layers:
                layer.trainable = True

            import pickle
            model_params = pickle.load(open( model_param_path, "rb" ))
            # Update the callback instance
            model_params = model_params
        else:
            model = mode_builder()
            MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())       

            
        from calbakLearningRateSchedulerPerBatch import calbakLearningRateSchedulerPerBatch
        lr_decay_callback = calbakLearningRateSchedulerPerBatch(model_params, None)
    
        # Modified Checkpoint Callback
        from calbakModelCheckpointEnhanced import calbakModelCheckpointEnhanced
        ckpt_callback = calbakModelCheckpointEnhanced(model_params = model_params, filepath=MODEL_PATH_CONSTANTS.GetCheckPointPath(), monitor='val_loss',
                                    save_best_only=True ,model_params_filepath=MODEL_PATH_CONSTANTS.GetModelParamPath() )
    
        callbacks  = callbacks + [lr_decay_callback, ckpt_callback]

        if get_model:
            return model

        # Start/resume training
        model.fit(train_x, train_y,
        batch_size = 32,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        validation_data=(valid_x, valid_y),
        callbacks=callbacks)


    def PrintModel(self, model):
        print(model.summary())
        print(f"model  Name : {(model.layers[1].get_weights())}")
        from tensorflow.keras.utils import plot_model
        MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())
        plot_model(model, to_file=MODEL_PATH_CONSTANTS.GetModelVizPath(), 
        show_shapes=True, show_layer_names=True, rankdir="LR")

    def evaluate_model(self, checkpoint_path, test_x, test_y):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)
        print(f"Evaluated Result : {model.evaluate(test_x, test_y)}")


    def model_predict(self, checkpoint_path, test_x):
        from tensorflow.keras.models import load_model

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path, custom_objects={"lr": self.get_lr_metric(optimizer)})
        y_pred = model.predict(test_x)
        print(f"Predicted value : {y_pred}")
        return y_pred


rnn_test_class = rnn_test()



from CommonConstants import MODEL_PATH_CONSTANTS


#region Check point with variable learning rate

Total_Epochs = 20
Initiate_at_Epoch = 20
Initiate_at_Loss = 0.04

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)



#Series prediction suing single value predictor

# # train_x, train_y, validation_x, validation_y , test_x, test_y = rnn_test_class.get_random_datasets(10000, 50, 1)

# rnn_test_class.fit_model(train_x ,train_y,  validation_x,validation_y , None, None, Total_Epochs, Initiate_at_Epoch, 1)
# rnn_test_class.model_predict(CheckPointPath,  test_x[:1, :])
# print(f"Actual value : {test_y[0]}")

# train_x, train_y, validation_x, validation_y , test_x, test_y = rnn_test_class.get_random_datasets(10000, 50, 10)
# rnn_test_class.predict_next_series(CheckPointPath,  test_x[:1, :], 10)
# print(f"Actual value : {test_y[0]}")
#endregion



#Series prediction using multi value predictor

# train_x, train_y, validation_x, validation_y , test_x, test_y = rnn_test_class.get_random_datasets(10000, 50, 10)

# rnn_test_class.fit_model(rnn_test_class.Build_Keras_rnn_multi_out_Model, train_x ,train_y,  validation_x,validation_y , None, None, Total_Epochs, Initiate_at_Epoch, 1)
# rnn_test_class.model_predict(CheckPointPath,  test_x[:1, :])
# print(f"Actual value : {test_y[0]}")
#endregion


#Series prediction using seq-to-seq model

# train_x, train_y, validation_x, validation_y , test_x, test_y = rnn_test_class.get_random_time_series_datasets(10000, 50, 10)

# rnn_test_class.fit_model(rnn_test_class.Build_Keras_rnn_Seq_to_Seq_1D_Conv_Model, train_x ,train_y[:, 3::2],  validation_x,validation_y[:, 3::2] , None, None, Total_Epochs, Initiate_at_Epoch, 1)
# y_pred = rnn_test_class.model_predict(CheckPointPath,  test_x[:1, :])
# print(f"Predicted value : {y_pred[:, -1]}")
# print(f"Actual value : {test_y[:1, -1]}")
# print(f"Mean Squared Error : {keras.metrics.mean_squared_error(test_y[:1, -1], y_pred[:, -1])}")

# rnn_test_class.auto_predict_1D_Conv_series(CheckPointPath,  test_x[:1, -1], 10)
# print(f"Actual value : {test_y[0]}")
#endregion



#Series prediction using 1DConv model

# train_x, train_y, validation_x, validation_y , test_x, test_y = rnn_test_class.get_random_time_series_datasets(10000, 50, 10)

# rnn_test_class.fit_model(rnn_test_class.Build_Keras_rnn_Seq_to_Seq_1D_Conv_Model, train_x ,train_y[:, 3::2],  validation_x,validation_y[:, 3::2] , None, None, Total_Epochs, Initiate_at_Epoch, 1)
# y_pred = rnn_test_class.model_predict(CheckPointPath,  test_x[:1, :])
# print(f"Predicted value : {y_pred[:, -1]}")
# print(f"Actual value : {test_y[:1, -1]}")
# print(f"Mean Squared Error : {keras.metrics.mean_squared_error(test_y[:1, -1], y_pred[:, -1])}")

# print(f"test_x : {test_x[:1]}")
# rnn_test_class.auto_predict_1D_Conv_series(CheckPointPath,  test_x[:1 , -1], 10)
# print(f"Actual value : {test_y[0]}")
#endregion

#Series prediction using WavNet model

train_x, train_y, validation_x, validation_y , test_x, test_y = rnn_test_class.get_random_time_series_datasets(10000, 50, 10)

# rnn_test_class.fit_model(rnn_test_class.Build_Keras_WAVNET_Model, train_x ,train_y,  validation_x,validation_y , None, None, Total_Epochs, Initiate_at_Epoch, 1)
# y_pred = rnn_test_class.model_predict(CheckPointPath,  test_x[:1, :])
# print(f"Predicted value : {y_pred[:, -1]}")
# print(f"Actual value : {test_y[:1, -1]}")
# print(f"Mean Squared Error : {keras.metrics.mean_squared_error(test_y[:1, -1], y_pred[:, -1])}")

# print(f"test_x : {test_x[:1]}")
import numpy as np
rnn_test_class.auto_predict_WavNet_series(CheckPointPath, np.expand_dims(test_x[:1 , -1], axis=0)  , 10)
# print(f"Actual value : {test_y[0]}")
#endregion