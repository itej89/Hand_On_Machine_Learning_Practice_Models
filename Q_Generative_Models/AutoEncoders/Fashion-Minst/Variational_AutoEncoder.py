import os, sys

from pathlib import Path

from tensorflow.keras import models
from tensorflow.python.keras import activations
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



    def Build_Auto_Encoder_PCA_UnderComplete(self):
        import tensorflow as tf
        from tensorflow import keras

        
        codings_size = 10

        inputs = keras.layers.Input(shape=[28, 28])
        z = keras.layers.Flatten()(inputs)
        z = keras.layers.Dense(150, activation = "selu")(z)
        z = keras.layers.Dense(100, activation = "selu")(z)

        coding_mean = keras.layers.Dense(codings_size)(z)
        coding_log_var = keras.layers.Dense(codings_size)(z)
        from Sampling import Sampling
        codings = Sampling()([coding_mean, coding_log_var])

        variational_Encoder = keras.Model(
            inputs=[inputs], outputs = [coding_mean, coding_log_var, codings])

        decoder_inputs = keras.layers.Input(shape=[codings_size])
        x = keras.layers.Dense(100, activation = "selu")(decoder_inputs)
        x = keras.layers.Dense(150, activation= "selu")(x)
        x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
        outputs = keras.layers.Reshape([28 ,28])(x)
        
        variational_Decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])

        _, _ , codings = variational_Encoder(inputs)
        reconstructions = variational_Decoder(codings)
        
         
        variational_ae = keras.Model(inputs=[inputs], outputs=[reconstructions])

        K = keras.backend

        latent_loss = -0.5 * K.sum(
        1 + coding_log_var - K.exp(coding_log_var) -
        K.square(coding_mean),
        axis=-1)
        variational_ae.add_loss(K.mean(latent_loss) / 784.)
        variational_ae.compile(loss="binary_crossentropy", optimizer="rmsprop")
        
        return variational_ae

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
        from tensorflow.keras.utils import plot_model
        MODEL_PATH_CONSTANTS.CreatePath(MODEL_PATH_CONSTANTS.GetDirectoryPath())
        plot_model(model, to_file=MODEL_PATH_CONSTANTS.GetModelVizPath(), 
        show_shapes=True, show_layer_names=True, rankdir="LR")


    def evaluate_model(self, checkpoint_path, test):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)
        print(f"Evaluated Result : {model.evaluate(test, test)}")


    def model_reconstruct(self, checkpoint_path, codings):
        from tensorflow.keras.models import load_model
        from Sampling import Sampling
        model = load_model(checkpoint_path, custom_objects={"Sampling": Sampling})
        self.PrintModel(model)
        variational_decoder = model.layers[2]
        images = variational_decoder(codings).numpy()
        return images

        

   


fmnist_model = fashion_mnist_model() 
train_x , train_y, test_x, test_y, validation_x, validation_y = fmnist_model.get_data_sets()


input_pipeline = fmnist_model.create_pipeline()

train_x = fmnist_model.runthrough_pipeline(input_pipeline, train_x)
validation_x = fmnist_model.runthrough_pipeline(input_pipeline, validation_x)
test_x = fmnist_model.runthrough_pipeline(input_pipeline, test_x)


from CommonConstants import MODEL_PATH_CONSTANTS

Total_Epochs = 500
Batch_Size = 32
Initiate_at_Epoch = 18
Initiate_at_Loss = 0.28

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)


# fmnist_model.fit_model(fmnist_model.Build_Auto_Encoder_PCA_UnderComplete, train_x , validation_x, CheckPointPath, ModleParamtPath, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)
codings_size = 10
import tensorflow as tf
codings = tf.random.normal(shape=[12, codings_size])
generated_images = fmnist_model.model_reconstruct(CheckPointPath, codings)

from PlotData import PlotData
objPlotData = PlotData()
objPlotData.shaow_images(generated_images, 12)

# test_2D = fmnist_model.model_DimensionalityReduction(CheckPointPath, test_x )
# from PlotData import PlotData
# objPlotData = PlotData()
# objPlotData.plot_scatter(test_2D, test_y)