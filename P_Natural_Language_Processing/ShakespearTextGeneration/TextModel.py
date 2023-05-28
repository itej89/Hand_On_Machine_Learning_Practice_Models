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



class TextModel :

    from LoadData import  Tokenizer
    tokenizer = Tokenizer()

    def get_datasets(self):
        from LoadData import LoadData
        objLoadData = LoadData()
        strData = objLoadData.load_Data(DATASET_PATH_CONSTANTS.GetBlobPath())

        encoded_data = self.tokenizer.Tokenize(strData)


        from Splitdata import SplitData
        split_data = SplitData()
        train, test, valid  = split_data.split_text(encoded_data, 75, 15, 10)


        return train, test, valid


    def preprocess(self,windows):
        X_batch, Y_batch = (windows[:, :-1], windows[:, 1:])
        return (tf.one_hot(X_batch, depth=self.tokenizer.max_id),Y_batch)

    def create_tf_dataset(self, data_set, repeat=None, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):
        import tensorflow as tf
        tfDataSet = tf.data.Dataset.from_tensor_slices(data_set)
        n_steps = 100
        window_length = n_steps + 1 
        tfDataSet = tfDataSet.window(window_length, shift=1, drop_remainder=True)
        tfDataSet = tfDataSet.flat_map(lambda window: window.batch(window_length))
        tfDataSet = tfDataSet.shuffle(shuffle_buffer_size)
        tfDataSet = tfDataSet.batch(batch_size)
        tfDataSet = tfDataSet.map(self.preprocess)
        return tfDataSet.prefetch(1)



    def Build_Keras_rnn_Seq_to_Seq_Model(self):
        model = keras.models.Sequential([
            # keras.layers.GRU(128, return_sequences=True ,input_shape=[None, self.tokenizer.max_id], dropout=0.2, recurrent_dropout=0.2),
            # keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            #Recurrent Dropout not supported in CUDNN
            keras.layers.GRU(128, return_sequences=True ,input_shape=[None, self.tokenizer.max_id], dropout=0.2, recurrent_dropout=0),
            keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0),
            keras.layers.TimeDistributed(keras.layers.Dense(self.tokenizer.max_id, activation="softmax"))
        ])

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        model.compile(loss="sparse_categorical_crossentropy",
        optimizer=optimizer)

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




    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


    def fit_model(self, mode_builder, train, valid, checkpoint_path=None, model_param_path=None, no_epochs=30, initial_epoch = 0, verbosity=1, get_model = False, batch_size=32,):
        
        from Model_Params import Model_Params
        model_params = Model_Params() 

        from calbakTensorBoard import calbakTensorboard
        from calbakEarlyStopping import calbakEarlyStopping
        callbacks = [calbakTensorboard.getTensorboardCalbak(MODEL_PATH_CONSTANTS().GetLogsPath()),
        calbakEarlyStopping.getEarlyStoppingCalbak()]


        # Load checkpoint:
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            # Load model:
            from tensorflow.keras.models import load_model
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
        model.fit(train,
        # steps_per_epoch = (836445//batch_size),  # number of training characters - 100(Since trainign set is generated by shifting be one character)
        steps_per_epoch = 1,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        validation_data=valid,
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
        model = load_model(checkpoint_path)
        y_pred = model.predict_classes(test_x)
        # print(f"Predicted value : {y_pred}")
        return y_pred

    def preprocess_new_setense(self, texts):
        import numpy as np
        X = np.array(objTextModel.tokenizer.tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, objTextModel.tokenizer.max_id)

    def next_char(self, model, text, temperature=1):
        X_new = self.preprocess_new_setense([text])
        y_proba = model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return self.tokenizer.tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text(self, text, checkpoint_path, n_chars=50, temperature=1):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)
        for _ in range(n_chars):
            text += self.next_char(model, text, temperature)
        return text

objTextModel = TextModel()



from CommonConstants import MODEL_PATH_CONSTANTS


#region Check point with variable learning rate

Total_Epochs = 10000
Initiate_at_Epoch = 0
Initiate_at_Loss = 0

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)




#Series prediction using seq-to-seq model

train, test, valid = objTextModel.get_datasets()
train_dataset = objTextModel.create_tf_dataset(train)
test_dataset = objTextModel.create_tf_dataset(test)
valid_dataset = objTextModel.create_tf_dataset(valid)

objTextModel.fit_model(objTextModel.Build_Keras_rnn_Seq_to_Seq_Model, train_dataset ,valid_dataset, CheckPointPath, ModleParamtPath, Total_Epochs, Initiate_at_Epoch, 1)

# full_Sentence = objTextModel.complete_text("How are yo", CheckPointPath,  50, 1)
# print(f"Predicted Sentence : {full_Sentence}")
# print(f"Predicted value : {y_pred[:, -1]}")
# print(f"Actual value : {test_y[:1, -1]}")
# print(f"Mean Squared Error : {keras.metrics.mean_squared_error(test_y[:1, -1], y_pred[:, -1])}")

# objTextModel.auto_predict_1D_Conv_series(CheckPointPath,  test_x[:1, -1], 10)
# print(f"Actual value : {test_y[0]}")
#endregion


