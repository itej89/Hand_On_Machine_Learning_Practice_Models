import os, sys

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent, "KerasCallbacks"))
sys.path.append(os.path.join(pwd.parent, "PipelineTransformers"))
sys.path.append(os.path.join(pwd.parent, "CustomComponents"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS_CSV, LABELS_PKL

from custom_loss import HuberLoss
from custom_metric import HuberMetric

class fashion_mnist_model :


    def split_data(self, pandas_frame):
     
        from Splitdata import SplitData
        objSplitData = SplitData()

        strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")
        
        strat_train_set.reset_index(inplace = True)
        strat_train_set, strat_val_set = objSplitData.stratified_split_data(_panda_data_frame=strat_train_set, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")


        strat_train_set = strat_train_set.drop("income_category", axis=1)
        strat_test_set = strat_test_set.drop("income_category", axis=1)
        strat_val_set = strat_val_set.drop("income_category", axis=1)
        # strat_train_set = strat_train_set.drop("ocean_proximity", axis=1)
        # strat_test_set = strat_test_set.drop("ocean_proximity", axis=1)
        # strat_val_set = strat_val_set.drop("ocean_proximity", axis=1)

        strat_train_set_data = strat_train_set.drop("index", axis=1)
        strat_train_set_data = strat_train_set_data.drop(LABELS_PKL.TARGET, axis=1)
        strat_train_set_labels = strat_train_set[LABELS_PKL.TARGET]

        strat_test_set_data = strat_test_set.drop(LABELS_PKL.TARGET, axis=1)
        strat_test_set_labels = strat_test_set[LABELS_PKL.TARGET]

        strat_val_set_data = strat_val_set.drop("index", axis=1)
        strat_val_set_data = strat_val_set_data.drop(LABELS_PKL.TARGET, axis=1)
        strat_val_set_labels = strat_val_set[LABELS_PKL.TARGET]

        return strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels,strat_test_set_data, strat_test_set_labels


    def get_data_sets(self):
        from pathlib import Path
        
        DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)
        # data = objLoadData.load_csv_data(DATASET_PATH_CONSTANTS.GetCSVPath())

        import numpy as np

        data = objLoadData.categorize_value_column(data, LABELS_PKL.MEDIA_INCOME, "income_category", \
        _bins=[0.,1.5,3.0,4.5,6.0,np.inf], _labels=[1,2,3,4,5])
        

        strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels, strat_test_set_data, strat_test_set_labels = self.split_data(data)

        return strat_train_set_data, strat_train_set_labels, strat_test_set_data, strat_test_set_labels,strat_val_set_data, strat_val_set_labels

    
    def create_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        input_pipeline = Pipeline([
            ('std_Scaler', StandardScaler())
        ])

        return input_pipeline
    
    def runthrough_pipeline(self, full_pipeline, data_frame):
        data_formated = full_pipeline.fit_transform(data_frame)
        return data_formated

    def get_num_attributes(self, data_frame):
        num_attribs = list(data_frame)
        return num_attribs

    def get_all_features(self, full_pipeline, data_frame):
        num_attribs = self.get_num_attributes(data_frame)
        return num_attribs




    def Build_Keras_Model(self):
        import tensorflow as tf
        from tensorflow import keras

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(30, name="First_HLayer", activation="relu", input_shape=[8]))
        model.add(keras.layers.Dense(1, name="Output_HLayer"))


        model.compile(loss="mean_squared_error",
        optimizer="sgd", metrics=["MeanSquaredError"])
        
        # model.compile(loss=HuberLoss(2.0),
        # optimizer="sgd", metrics=[HuberMetric(2.0)])

        return model

    def Build_Wide_Deep_Model(self):
        import tensorflow as tf
        from tensorflow import keras

        input = keras.layers.Input(shape=[8])
        hidden1 = keras.layers.Dense(30, activation="relu")(input)
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.Concatenate()([input, hidden2])
        output = keras.layers.Dense(1)(concat)

        model = keras.models.Model(inputs=[input], outputs=[output])

        model.compile(loss="mean_squared_error",
        optimizer="sgd", metrics=["MeanSquaredError"])

        return model

    def Build_model_out_of_params(self, n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)

        from tensorflow import keras

        model = keras.models.Sequential()
        options={"input_shape":input_shape}

        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activatgio="relu", **options))
            options = {}
        
        model.add(keras.layers.Dense(1, **options))

        optimizer = keras.optimizers.SGD(learning_rate)
        model.compile(loss="mse", optimizer=optimizer)

        return model

    def Explore_Hyper_Params(self, train_X, train_y, X_valid, y_valid, checkpoint_path=None, model_param_path=None, no_epochs=30, batch_size=32, initial_epoch = 0, verbosity=1, get_model = False):
        
#region Build scikit regerssor from keras model
        from tensorflow import keras

        keras_reg = keras.wrappers.scikit_learn.KerasRegressor(self.Build_model_out_of_params)

        #Explore params
        from scipy.stats import reciprocal
        from sklearn.model_selection import RandomizedSearchCV

        import numpy as np
        param_distrib = {
            "n_hidden" : (0, 1 ,2, 3),
            "n_neurons": (30,40),
            "learning_rate": (0.001, 0.0005)
        }

        rnd_search_cv = RandomizedSearchCV(keras_reg, param_distrib, n_jobs=-1, n_iter=10, cv=3)
#endregion

#region Build callbacks
        from Model_Params import Model_Params
        model_params = Model_Params() 

        from calbakTensorBoard import calbakTensorboard
        from calbakEarlyStopping import calbakEarlyStopping
        callbacks = [calbakTensorboard.getTensorboardCalbak(MODEL_PATH_CONSTANTS().GetLogsPath()),
        calbakEarlyStopping.getEarlyStoppingCalbak()]

        MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())       

        # Modified Checkpoint Callback
        from calbakModelCheckpointEnhanced import calbakModelCheckpointEnhanced
        ckpt_callback = calbakModelCheckpointEnhanced(model_params = model_params, filepath=MODEL_PATH_CONSTANTS.GetCheckPointPath(), monitor='val_loss',
                                    save_best_only=True ,model_params_filepath=MODEL_PATH_CONSTANTS.GetModelParamPath() )
    
        callbacks  = callbacks + [ckpt_callback]
#endregion

#region Perform Param Search
        rnd_search_cv.fit(train_X, train_y,
        batch_size=batch_size,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks)
#endregion

        print(rnd_search_cv.best_params_)


    def fit_model(self, train_X, train_y, X_valid, y_valid, checkpoint_path=None, model_param_path=None, no_epochs=30, batch_size=32, initial_epoch = 0, verbosity=1, get_model = False):
        
       
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
            model = load_model(checkpoint_path, custom_objects={"HuberLoss": HuberLoss})

            import pickle
            model_params = pickle.load(open( model_param_path, "rb" ))
            # Update the callback instance
            model_params = model_params
        else:
            model = self.Build_Keras_Model()
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
        model.fit(train_X, train_y,
        batch_size=batch_size,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        validation_data=(X_valid, y_valid),
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
        print(f"Predicted value : {model.predict(test_x)}")



fmnist_model = fashion_mnist_model() 
train_x , train_y, test_x, test_y, validation_x, validation_y = fmnist_model.get_data_sets()


input_pipeline = fmnist_model.create_pipeline()

train_x = fmnist_model.runthrough_pipeline(input_pipeline, train_x)
validation_x = fmnist_model.runthrough_pipeline(input_pipeline, validation_x)
test_x = fmnist_model.runthrough_pipeline(input_pipeline, test_x)


from CommonConstants import MODEL_PATH_CONSTANTS


#region Check point with variable learning rate

Total_Epochs = 20
Batch_Size = 32 #"Friends don’t let friends use mini-batches larger than 32“
Initiate_at_Epoch = 0
Initiate_at_Loss = 0

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)

fmnist_model.fit_model(train_x , train_y, validation_x, validation_y, None, None, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)

fmnist_model.model_predict(CheckPointPath, test_x)

print(f"Actual value : {test_y}")
#endregion

#region hyper param explorer with scikit learn

# Total_Epochs = 2
# Batch_Size = 32 #"Friends don’t let friends use mini-batches larger than 32“
# Initiate_at_Epoch = 0
# Initiate_at_Loss = 0
# fmnist_model.Explore_Hyper_Params(train_x , train_y, validation_x, validation_y, None, None, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)

#endregion