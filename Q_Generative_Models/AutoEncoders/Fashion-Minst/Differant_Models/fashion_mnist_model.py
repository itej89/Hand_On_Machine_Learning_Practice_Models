import os, sys

from pathlib import Path 


sys.path.append(os.path.join(pwd.parent, "KerasCallbacks"))
sys.path.append(os.path.join(pwd.parent, "PipelineTransformers"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS

class fashion_mnist_model :



    def split_data(self, train_X, train_y):
        pass

    def get_data_sets(self):
        from pathlib import Path
        
        DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        train_x , train_y, text_x, test_y = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)

        from Splitdata import SplitData
        objSplitData = SplitData()

        train_x, train_y, validation_x, validation_y = objSplitData.split_train_to_validation(train_x, train_y)
        
        return train_x , train_y, text_x, test_y, validation_x, validation_y


    
    def create_pipeline(self):
        from sklearn.pipeline import Pipeline
        from StandardScaler3D import StandardScaler3D
        
        input_pipeline = Pipeline([
            ('std_Scaler', StandardScaler3D())
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
        model.add(keras.layers.Flatten(name="Input_Layer", input_shape=[28, 28]))
        model.add(keras.layers.Dense(300, name="First_HLayer", activation="relu"))
        model.add(keras.layers.Dense(100, name="Second_HLayer", activation="relu"))
        model.add(keras.layers.Dense(10, name="Output_Layer", activation="softmax"))

        model.compile(loss="sparse_categorical_crossentropy",
        optimizer="sgd", metrics=["accuracy"])

        return model


        

    def fit_model(self, train_X, train_y, X_valid, y_valid, checkpoint_path=None, model_param_path=None, no_epochs=30, batch_size=250, initial_epoch = 0, verbosity=1, get_model = False):
        
       
        from Model_Params import Model_Params
        model_params = Model_Params() 

        from calbakTensorBoard import calbakTensorboard
        callbacks = [calbakTensorboard.getTensorboardCalbak()]


        # Load checkpoint:
        if checkpoint_path is not None and os.path.exists(checkpoint_path) and os.path.exists(model_param_path):
            # Load model:
            from tensorflow.keras.models import load_model
            model = load_model(checkpoint_path)

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
    
        # callbacks  = callbacks + [lr_decay_callback, ckpt_callback]
        callbacks  = callbacks + [lr_decay_callback]

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
        print(f"Predicted Probabilities : {model.predict(test_x)}")

    def model_predict_class(self, checkpoint_path, test_x):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)
        print(f"Predicted Classess : {model.predict_classes(test_x)}")
   


fmnist_model = fashion_mnist_model() 
train_x , train_y, test_x, test_y, validation_x, validation_y = fmnist_model.get_data_sets()


input_pipeline = fmnist_model.create_pipeline()

train_x = fmnist_model.runthrough_pipeline(input_pipeline, train_x)
validation_x = fmnist_model.runthrough_pipeline(input_pipeline, validation_x)
test_x = fmnist_model.runthrough_pipeline(input_pipeline, test_x)


from CommonConstants import MODEL_PATH_CONSTANTS

Total_Epochs = 50
Batch_Size = 250
Initiate_at_Epoch = 0
Initiate_at_Loss = 0

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)

fmnist_model.fit_model(train_x , train_y, validation_x, validation_y, CheckPointPath, ModleParamtPath, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)

# fmnist_model.model_predict_class(CheckPointPath, test_x)