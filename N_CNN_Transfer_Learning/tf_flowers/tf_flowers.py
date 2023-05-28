from operator import add
from tensorflow.keras import activations
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



class tf_flowers_classifier :

#region DataProcessing

    #region To get datasets as tf dataset

    #region generate split files

    def split_pandas_frame_to_protobuf_files(self, pandas_frame, split_percent, filename_template):
        file_size = int(pandas_frame.shape[0]*split_percent/100)
        number_of_files = int(pandas_frame.shape[0]/file_size) if pandas_frame.shape[0]%file_size == 0 else int(pandas_frame.shape[0]/file_size)+1
        for i in range(number_of_files):
            df = pandas_frame[file_size*i:file_size*(i+1)]
            file_path = filename_template.format(i+1)
            BytesList = tf.train.BytesList
            FloatList = tf.train.FloatList
            Int64List = tf.train.Int64List
            Feature = tf.train.Feature
            Features= tf.train.Features
            Example = tf.train.Example
            with tf.io.TFRecordWriter(file_path) as f:
                for index, row in df.iterrows():
                    row_features = {}
                    import numpy as np
                    for header in df.columns:
                        value = row[header]
                        if type(value) is np.float64:
                            row_features[header] = Feature(float_list=FloatList(value=[value]))
                        if type(value) is int:
                            row_features[header] = Feature(int64_list=Int64List(value=[value]))
                        if type(value) is str:
                            row_features[header] = Feature(bytes_list=BytesList(value=[bytes(value, 'utf-8')]))
                    housing_entry  = Example(
                        features=Features(
                            feature=row_features
                        )
                    )
                    f.write(housing_entry.SerializeToString())

    def create_tf_proto_buf_files(self):

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data = objLoadData.load_Data(DATASET_PATH_CONSTANTS.GetBlobPath(), False)
        # data = objLoadData.load_csv_data(DATASET_PATH_CONSTANTS.GetCSVPath())

        import numpy as np
        data = objLoadData.categorize_value_column(data, LABELS_PKL.MEDIA_INCOME, "income_category", \
        _bins=[0.,1.5,3.0,4.5,6.0,np.inf], _labels=[1,2,3,4,5])

        strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels, strat_test_set_data, strat_test_set_labels = self.split_data(data)
        strat_train_set_data[LABELS_PKL.TARGET] =  strat_train_set_labels
        strat_val_set_data[LABELS_PKL.TARGET] =  strat_val_set_labels
        strat_test_set_data[LABELS_PKL.TARGET] =  strat_test_set_labels
        
        DATASET_PATH_CONSTANTS.RemovePath(DATASET_PATH_CONSTANTS.GetSplitDir())
        DATASET_PATH_CONSTANTS.CreatePath(DATASET_PATH_CONSTANTS.GetTrainSplitDir())
        DATASET_PATH_CONSTANTS.CreatePath(DATASET_PATH_CONSTANTS.GetTestSplitDir())
        DATASET_PATH_CONSTANTS.CreatePath(DATASET_PATH_CONSTANTS.GetValidationSplitDir())

        self.split_pandas_frame_to_protobuf_files(strat_train_set_data, 1, DATASET_PATH_CONSTANTS.GetTrainSplitPath())
        self.split_pandas_frame_to_protobuf_files(strat_val_set_data, 1, DATASET_PATH_CONSTANTS.GetValidationSplitPath())
        self.split_pandas_frame_to_protobuf_files(strat_test_set_data, 1, DATASET_PATH_CONSTANTS.GetTestSplitPath())

    def get_tf_list_file_dataset(self):

        return [DATASET_PATH_CONSTANTS.GetTrainSplitPath().format("*")], \
            [DATASET_PATH_CONSTANTS.GetTestSplitPath().format("*")], \
                [DATASET_PATH_CONSTANTS.GetValidationSplitPath().format("*")]

    def get_tfds_datasets(self, Fetch_Tuple = False):
        from LoadData import LoadData_TFDS
        objLoadData = LoadData_TFDS()
        import tensorflow_datasets as tfds 
        train_split, valid_split, test_split  = ["train[:75%]", "train[75%:90%]", "train[90%:]"]

        dataset_train = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), Fetch_Tuple, train_split)
        dataset_test = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), Fetch_Tuple, test_split)
        dataset_valid = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), Fetch_Tuple, valid_split)
       
        return dataset_train, dataset_test,dataset_valid

    #endregion

    #region preprocess data

    def preprocess(self, image, label):


        #Using feature_columns for example spec parsing
        # Template 
        # housing_median_age = tf.feature_column.numeric_column("housing_median_age", normalizer_fn=lambda x: (x - age_mean) / age_std)
        # bucketized_income = tf.feature_column.bucketized_column(median_income, boundaries=[1.5, 3., 4.5, 6.])

        # Categorical Features
        # ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        # ocean_proximity = tf.feature_column.categorical_column_with_vocabulary_list("ocean_proximity", ocean_prox_vocab)
        # city_hash = tf.feature_column.categorical_column_with_hash_bucket("city", hash_bucket_size=1000)

        # Crossed Categorical Features
        # bucketized_age = tf.feature_column.bucketized_column(housing_median_age, boundaries=[-1., -0.5, 0., 0.5, 1.]) # age was scaled
        # age_and_ocean_proximity = tf.feature_column.crossed_column([bucketized_age, ocean_proximity], hash_bucket_size=100)

        # Encoding Categorical Features Using One-Hot Vectors
        # latitude = tf.feature_column.numeric_column("latitude")
        # longitude = tf.feature_column.numeric_column("longitude")
        # bucketized_latitude = tf.feature_column.bucketized_column(latitude, boundaries=list(np.linspace(32., 42., 20 - 1)))
        # bucketized_longitude = tf.feature_column.bucketized_column(longitude, boundaries=list(np.linspace(-125., -114., 20 - 1)))
        # location = tf.feature_column.crossed_column([bucketized_latitude, bucketized_longitude], hash_bucket_size=1000)

        # Encoding Categorical Features Using One-Hot Vectors
        # ocean_proximity_one_hot = tf.feature_column.indicator_column(ocean_proximity)   

        # Encoding Categorical Features Using Embeddings
        # ocean_proximity_embed = tf.feature_column.embedding_column(ocean_proximity,dimension=2)  

        # columns = [bucketized_age, ....., median_house_value] # all features + target
        # feature_descriptions = tf.feature_column.make_parse_example_spec(columns)

        # def parse_examples(serialized_examples):
        #     examples = tf.io.parse_example(serialized_examples, feature_descriptions)
        #     targets = examples.pop("median_house_value") # separate the targets
        #     return examples, targets


        # image = tf.feature_column.numeric_column("image")
        # digit_vocab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # digit_label = tf.feature_column.categorical_column_with_vocabulary_list("label",  digit_vocab)
        # target = tf.feature_column.indicator_column(digit_label)
        # columns = [image, target]
        # print("Make Parse example spec")
        # feature_descriptions = tf.feature_column.make_parse_example_spec(columns)

        # print("Parse example spec : {}".format(serialized_record))
        # fields = tf.io.parse_example(serialized_record,feature_descriptions)
        # print(fields["image"])
        # print(fields["target"])



     

        resized_image = tf.image.resize(image, [224, 224])
        final_image = keras.applications.xception.preprocess_input(resized_image)
        return final_image, label



    def create_tfds_runtime_dataset(self, filepaths, repeat=None, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):
        dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
        dataset = dataset.interleave(
            lambda filepath:  tf.data.TFRecordDataset(filepath).skip(1),
            cycle_length=n_readers, num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.preprocess_pikle_set, num_parallel_calls=n_parse_threads)
        return dataset.prefetch(1)

    def create_Datasets_from_files(self, filepaths):
        dataset = tf.data.Dataset.list_files(filepaths)
        return dataset

    def create_tfds_train_dataset_streams(self, dataset, repeat=None, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):
        dataset = dataset.repeat(repeat).shuffle(shuffle_buffer_size)
        dataset = dataset.map(self.preprocess, num_parallel_calls=n_parse_threads).batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    #endregion

    #endregion

#endregion


    def Build_Keras_Sequential_Model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(30, name="First_HLayer", activation="relu", input_shape=[768]))
        model.add(keras.layers.Dense(30, name="Second_HLayer", activation="relu"))
        model.add(keras.layers.Dense(10, name="Output_Layer", activation="softmax"))


        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="sparse_categorical_crossentropy",
        optimizer=optimizer, metrics=["accuracy", lr_metric])

        return model

    def Build_Resnet_34_Model(self):
        from ResidualUnit import ResidualUnit
        from ResidualUnit import DefaultConv2D
        model = keras.Sequential()
        model.add(DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224,224,3]))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
        prev_filters = 64
        for filters in [64] * 3 + [128]  *4 + [256] * 6 + [512]  * 3:
            strides = 1 if filters == prev_filters else 2
            model.add(ResidualUnit(filters, strides = strides))
            prev_filters = filters 
        model.add(keras.layers.GlobalAvgPool2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation="softmax"))

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="sparse_categorical_crossentropy",
        optimizer=optimizer, metrics=["accuracy", lr_metric])

        return model

    def Build_Xception_transfer_learning_Model(self):
        base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
        avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(10, activation = "softmax")(avg)

        model = keras.models.Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)

        lr_metric = self.get_lr_metric(optimizer)
        model.compile(loss="sparse_categorical_crossentropy",
        optimizer=optimizer, metrics=["accuracy", lr_metric])

        return model



    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


    def fit_model(self, train, valid, checkpoint_path=None, model_param_path=None, no_epochs=30, initial_epoch = 0, verbosity=1, get_model = False):
        
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
            model = self.Build_Xception_transfer_learning_Model()
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
        from ResidualUnit import ResidualUnit
        model = load_model(checkpoint_path, custom_objects={'ResidualUnit': ResidualUnit})
        print(f"Evaluated Result : {model.evaluate(test_x, test_y)}")


    def model_predict(self, checkpoint_path, test_x):
        from tensorflow.keras.models import load_model
        from ResidualUnit import ResidualUnit

        optimizer = keras.optimizers.Nadam(0.001, beta_1=0.9, beta_2=0.999)
        model = load_model(checkpoint_path, custom_objects={"lr": self.get_lr_metric(optimizer)})
        print(f"Predicted value : {model.predict(test_x)}")
        for x, y in test_x:
            print(f"Actual value : {y}")



tf_flowers_class = tf_flowers_classifier()

train, test, validation = tf_flowers_class.get_tfds_datasets(Fetch_Tuple = True)
train_set = tf_flowers_class.create_tfds_train_dataset_streams(train, repeat=1, batch_size=32)
test_set = tf_flowers_class.create_tfds_train_dataset_streams(test, repeat=1, batch_size=1)
validation_set = tf_flowers_class.create_tfds_train_dataset_streams(validation, repeat=1, batch_size=32)


from CommonConstants import MODEL_PATH_CONSTANTS


#region Check point with variable learning rate

Total_Epochs = 20
Initiate_at_Epoch = 20
Initiate_at_Loss = 0.30

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)

# tf_flowers_class.fit_model(train_set , validation_set, None, None, Total_Epochs, Initiate_at_Epoch, 1)

tf_flowers_class.model_predict(CheckPointPath, test_set.take(1))

# print(f"Actual value : {test_y}")
#endregion

#region hyper param explorer with scikit learn

# Total_Epochs = 2
# Batch_Size = 32 #"Friends don’t let friends use mini-batches larger than 32“
# Initiate_at_Epoch = 0
# Initiate_at_Loss = 0
# housing_model.Explore_Hyper_Params(train_x , train_y, validation_x, validation_y, None, None, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)

#endregion


   #Keras helper funciton for preprocessing images from a direcotry and generating tensorflow dataset
        # tf.keras.preprocessing.image_dataset_from_directory(
        #     directory,
        #     labels="inferred",
        #     label_mode="int",
        #     class_names=None,
        #     color_mode="rgb",
        #     batch_size=32,
        #     image_size=(256, 256),
        #     shuffle=True,
        #     seed=None,
        #     validation_split=None,
        #     subset=None,
        #     interpolation="bilinear",
        #     follow_links=False,
        #     smart_resize=False,
        # )