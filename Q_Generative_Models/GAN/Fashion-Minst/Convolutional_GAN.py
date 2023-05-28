import os, sys
from re import A


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
        train_x , train_y, test_x, test_y = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)

        train_x = train_x.reshape(-1, 28, 28, 1) * 2. - 1.

        test_x = test_x.reshape(-1, 28, 28, 1) * 2. - 1.
        return train_x, test_x
    
 
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

        # from Splitdata import SplitData
        # objSplitData = SplitData()

        # train_x, train_y, validation_x, validation_y = objSplitData.split_train_to_validation(train_x, train_y)
        


        # return train_x , train_y, text_x, test_y, validation_x, validation_y

    def normalize(self, data):
        data = tf.cast(data, tf.float32, name=None)
        # mean = tf.math.reduce_mean(data)
        # std = tf.math.reduce_std(data)
        # data = tf.subtract(data, mean)
        # data = tf.divide(data, std)
        return data


    def create_tfds_runtime_dataset(self, data, repeat=None, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):

        dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(self. normalize)
        dataset = dataset.prefetch(1)
        return dataset
    


    def get_num_attributes(self, data_frame):
        num_attribs = list(data_frame)
        return num_attribs

    def get_all_features(self, full_pipeline, data_frame):
        num_attribs = self.get_num_attributes(data_frame)
        return num_attribs



    def Build_GAN(self, codings_size):

        generator = keras.models.Sequential([
            keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
            keras.layers.Reshape([7, 7 , 128]),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="selu"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh")
        ])

        discriminator = keras.models.Sequential([
            keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2),
                                    input_shape=[28, 28, 1]),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2)),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        gan = keras.models.Sequential([generator, discriminator])

        # discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
        # discriminator.trainable = False

        # gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
        self.PrintModel(gan)
        return gan

        
            

    def fit_model(self, Build_Model, train, checkpoint_path=None, model_param_path=None, codings_size = 30, no_epochs=30, batch_size=250, initial_epoch = 0, verbosity=1, get_model = False):
        
       
        from Exp_Sched_Params import Exp_Sched_Params
        model_params = Exp_Sched_Params()

        from calbakTensorBoard import calbakTensorboard
        callbacks = [calbakTensorboard.getTensorboardCalbak()]


        # Load checkpoint:
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            # Load model:
            from tensorflow.keras.models import load_model
            from tensorflow import keras
            model = load_model(checkpoint_path, custom_objects={
        'LeakyReLU': keras.layers.LeakyReLU
    },)

        else:
            model = Build_Model(codings_size)
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
        generator, discriminator = model.layers
        discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
        discriminator.trainable = False

        model.compile(loss="binary_crossentropy", optimizer="rmsprop")
        reconstructions = []

        for epoch in range(initial_epoch, Total_Epochs):
            generated_images = None
            batch_count = 0
            loss_metric = {}
            for X_batch in train:
                console_string = f"Epoch : {epoch+1}/{Total_Epochs} -> Batch : {batch_count}"
                print('\x1b[2K\r', end='\r')
                tf.print(console_string, end='\r')
                batch_count = batch_count+1
                # phase 1 - training the discriminator
                noise = tf.random.normal(shape=[batch_size, codings_size])
                generated_images = generator(noise)
                # reconstructions = generated_images
                X_fake_and_real = tf.concat([generated_images, X_batch],
                axis=0)
                y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
                discriminator.trainable = True  
                discriminator.train_on_batch(X_fake_and_real, y1)
                # phase 2 - training the generator
                noise = tf.random.normal(shape=[batch_size, codings_size])
                y2 = tf.constant([[1.]] * batch_size)
                discriminator.trainable = False
                loss_metrics = model.train_on_batch(noise, y2, return_dict=True)
                loss_metric = loss_metrics
            

            # from PlotData import PlotData
            # objPlotData = PlotData()
            # objPlotData.shaow_images(reconstructions, batch_size)
                
            keras.models.save_model(model, MODEL_PATH_CONSTANTS.GetCheckPointPath().format(epoch=epoch + 1, val_loss=loss_metric["loss"]))


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


    def model_reconstruct(self, checkpoint_path, n_images=5):
        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path, custom_objects={
        'LeakyReLU': keras.layers.LeakyReLU
    },)
        generator, discriminator = model.layers
        self.PrintModel(generator)
        codings_size = 100
        noise = tf.random.normal(shape=[n_images, codings_size])
        generated_images = generator(noise)
        return generated_images


        

   


fmnist_model = fashion_mnist_model() 
train , test = fmnist_model.get_data_sets()

input_pipeline = fmnist_model.create_pipeline()
train = fmnist_model.runthrough_pipeline(input_pipeline, train)
test = fmnist_model.runthrough_pipeline(input_pipeline, test)

train_dataset = fmnist_model.create_tfds_runtime_dataset(train)
test_dataset = fmnist_model.create_tfds_runtime_dataset(test)


from CommonConstants import MODEL_PATH_CONSTANTS

Total_Epochs = 50
Batch_Size = 32
Initiate_at_Epoch = 50
Initiate_at_Loss = 0.00

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)


# fmnist_model.fit_model(fmnist_model.Build_GAN, train_dataset , CheckPointPath, ModleParamtPath, 100, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)


reconstructions = fmnist_model.model_reconstruct(CheckPointPath ,10)

from PlotData import PlotData
objPlotData = PlotData()
objPlotData.shaow_images(reconstructions, 10)

# test_2D = fmnist_model.model_DimensionalityReduction(CheckPointPath, test_x )
# from PlotData import PlotData
# objPlotData = PlotData()
# objPlotData.plot_scatter(test_2D, test_y)