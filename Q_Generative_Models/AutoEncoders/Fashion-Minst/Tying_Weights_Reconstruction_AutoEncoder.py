import os, sys

from pathlib import Path

from tensorflow.keras import models
from tensorflow.python.keras.engine.sequential import Sequential 

pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent, "KerasCallbacks"))
sys.path.append(os.path.join(pwd.parent, "Dropout"))
sys.path.append(os.path.join(pwd.parent, "PipelineTransformers"))
sys.path.append(os.path.join(pwd.parent, "CustomComponents"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS


import matplotlib.pyplot as plt

# from custom_loss import HuberLoss

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
        from Normalize3D import Normalize3D
        
        input_pipeline = Pipeline([
            ('std_Scaler', Normalize3D())
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


    def Build_Auto_Encoder_Weights_Tied_UnderComplete(self):
        import tensorflow as tf
        from tensorflow import keras

        dense_1 = keras.layers.Dense(100, activation="selu")
        dense_2 = keras.layers.Dense(30, activation="selu")

        encoder = keras.models.Sequential(name= "encoder")
        encoder.add(keras.layers.Flatten(name="Input_Layer", input_shape=[28, 28]))
        encoder.add(dense_1)
        encoder.add(dense_2)
        
        from DenseTranspose import DenseTranspose
        decoder = keras.models.Sequential(name= "decoder")
        decoder.add(DenseTranspose(dense_2 ,activation_name = "selu"))
        decoder.add(DenseTranspose(dense_1 ,activation_name = "sigmoid"))
        decoder.add(keras.layers.Reshape([28 , 28]))
       

        stacked_ae = keras.models.Sequential([encoder, decoder])
        stacked_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5))
        
        return stacked_ae


    def fit_model(self, Build_Model, train, valid, checkpoint_path=None, model_param_path=None, no_epochs=30, batch_size=250, initial_epoch = 0, verbosity=1, get_model = False):
        
       
        from Exp_Sched_Params import Exp_Sched_Params
        model_params = Exp_Sched_Params()

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
            model = Build_Model()
            MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())       

            
        from calbakLearningRateExponentialSchedulerPerEpoch import calbakLearningRateExponentialSchedulerPerEpoch
        lr_decay_callback = calbakLearningRateExponentialSchedulerPerEpoch(model_params, None)
    
        # Modified Checkpoint Callback
        from calbakModelCheckpointEnhanced import calbakModelCheckpointEnhanced
        ckpt_callback = calbakModelCheckpointEnhanced(model_params = model_params, filepath=MODEL_PATH_CONSTANTS.GetCheckPointPath(), monitor='val_loss',
                                    save_best_only=True ,model_params_filepath=MODEL_PATH_CONSTANTS.GetModelParamPath() )
    
        # callbacks  = callbacks + [lr_decay_callback, ckpt_callback]

        from tensorflow import keras
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
        lr_perf_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

        callbacks  = callbacks + [lr_perf_scheduler, early_stopping_cb, ckpt_callback]

        if get_model:
            return model

        # Start/resume training
        model.fit(train, train,
        batch_size=batch_size,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        validation_data=(valid, valid),
        callbacks=callbacks)


    def PrintModel(self, model):
        print(model.summary())
        print(f"model  Name : {(model.layers[1].get_weights())}")
        from tensorflow.keras.utils import plot_model
        MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())
        plot_model(model, to_file=MODEL_PATH_CONSTANTS.GetModelVizPath(), 
        show_shapes=True, show_layer_names=True, rankdir="LR")


    def evaluate_model(self, checkpoint_path, test):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)
        print(f"Evaluated Result : {model.evaluate(test, test)}")


    def model_reconstruct(self, checkpoint_path, test_x,  n_images=5):
        from tensorflow.keras.models import load_model
        from DenseTranspose import DenseTranspose
        model = load_model(checkpoint_path, custom_objects={"DenseTranspose": DenseTranspose})
        reconstructions = model.predict(test_x[:n_images])
        return reconstructions

    def model_DimensionalityReduction(self, checkpoint_path, test):
        from tensorflow.keras.models import load_model
        from DenseTranspose import DenseTranspose
        model = load_model(checkpoint_path, custom_objects={"DenseTranspose": DenseTranspose})

        #Extract encoder and decoder models
        encoder = model.layers[0]
        decoder = model.layers[1]

        decoder.build(input_shape=(None, 30))


        #Perform dimentionality reduction
        from sklearn.manifold import TSNE
        x_compressed = encoder.predict(test)
        tsne = TSNE()
        X_valid_2D = tsne.fit_transform(x_compressed)
        return X_valid_2D

        

   


fmnist_model = fashion_mnist_model() 
train_x , train_y, test_x, test_y, validation_x, validation_y = fmnist_model.get_data_sets()


input_pipeline = fmnist_model.create_pipeline()

train_x = fmnist_model.runthrough_pipeline(input_pipeline, train_x)
validation_x = fmnist_model.runthrough_pipeline(input_pipeline, validation_x)
test_x = fmnist_model.runthrough_pipeline(input_pipeline, test_x)


from CommonConstants import MODEL_PATH_CONSTANTS

Total_Epochs = 50
Batch_Size = 32
Initiate_at_Epoch = 28
Initiate_at_Loss = 0.26

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)


# fmnist_model.fit_model(fmnist_model.Build_Auto_Encoder_Weights_Tied_UnderComplete, train_x , validation_x, CheckPointPath, ModleParamtPath, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)


reconstructions = fmnist_model.model_reconstruct(CheckPointPath, test_x ,50)

from PlotData import PlotData
objPlotData = PlotData()
objPlotData.show_reconstructions(reconstructions, test_x, 50)

# test_2D = fmnist_model.model_DimensionalityReduction(CheckPointPath, test_x )
# from PlotData import PlotData
# objPlotData = PlotData()
# objPlotData.plot_scatter(test_2D, test_y)