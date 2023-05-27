import os, sys

from pathlib import Path

from tensorflow.python.ops.gen_image_ops import NonMaxSuppression 


pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent, "KerasCallbacks"))
sys.path.append(os.path.join(pwd.parent, "PipelineTransformers"))
sys.path.append(os.path.join(pwd.parent, "CustomComponents"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS_CSV, LABELS_PKL

from custom_loss import HuberLoss
from custom_metric import HuberMetric

class california_housing :

#region DataProcessing
    def split_data(self, pandas_frame):
     
        from Splitdata import SplitData
        objSplitData = SplitData()

        strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")
        
        strat_train_set.reset_index(inplace = True)
        strat_train_set, strat_val_set = objSplitData.stratified_split_data(_panda_data_frame=strat_train_set, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")


        strat_train_set = strat_train_set.drop("income_category", axis=1)
        strat_test_set = strat_test_set.drop("income_category", axis=1)
        strat_val_set = strat_val_set.drop("income_category", axis=1)

        strat_train_set_data = strat_train_set.drop("index", axis=1)
        strat_train_set_data = strat_train_set_data.drop(LABELS_PKL.TARGET, axis=1)
        strat_train_set_labels = strat_train_set[LABELS_PKL.TARGET]

        strat_test_set_data = strat_test_set.drop(LABELS_PKL.TARGET, axis=1)
        strat_test_set_labels = strat_test_set[LABELS_PKL.TARGET]

        strat_val_set_data = strat_val_set.drop("index", axis=1)
        strat_val_set_data = strat_val_set_data.drop(LABELS_PKL.TARGET, axis=1)
        strat_val_set_labels = strat_val_set[LABELS_PKL.TARGET]

        print(type(strat_val_set_data))
        print(type(strat_val_set_labels))

        return strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels,strat_test_set_data, strat_test_set_labels

#region To get datasets as numpy arrays
    def get_data_sets(self):
        from pathlib import Path
        pwd = Path(os.path.abspath(__file__))
        DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)

        import numpy as np
        data = objLoadData.categorize_value_column(data, "MedInc", "income_category", \
        _bins=[0.,1.5,3.0,4.5,6.0,np.inf], _labels=[1,2,3,4,5])

        strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels, strat_test_set_data, strat_test_set_labels = self.split_data(data)

        return strat_train_set_data, strat_train_set_labels, strat_test_set_data, strat_test_set_labels,strat_val_set_data, strat_val_set_labels
#endregion

    #region To get datasets as tf dataset

    #region generate split files

    def split_pandas_frame_to_protobuf_files(self, pandas_frame, split_percent, filename_template):
        file_size = int(pandas_frame.shape[0]*split_percent/100)
        number_of_files = int(pandas_frame.shape[0]/file_size) if pandas_frame.shape[0]%file_size == 0 else int(pandas_frame.shape[0]/file_size)+1
        for i in range(number_of_files):
            df = pandas_frame[file_size*i:file_size*(i+1)]
            file_path = filename_template.format(i+1)
            import tensorflow as tf
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

    #endregion

    #region preprocess data

    def preprocess_pikle_set(self, serialized_record):

        import tensorflow as tf

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

        feature_description = {
        "MedInc": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "HouseAge": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "AveRooms": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "AveBedrms": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "Population": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "AveOccup": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "Latitude": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "Longitude": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "target": tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
        }

        fields = tf.io.parse_example(serialized_record,feature_description)

        std_median_income = (fields["MedInc"] - tf.reduce_mean(fields["MedInc"]))/tf.math.reduce_std(fields["MedInc"])
        std_housing_median_age = (fields["HouseAge"] - tf.reduce_mean(fields["HouseAge"]))/tf.math.reduce_std(fields["HouseAge"])
        std_total_rooms = (fields["AveRooms"] - tf.reduce_mean(fields["AveRooms"]))/tf.math.reduce_std(fields["AveRooms"])
        std_total_bedrooms = (fields["AveBedrms"] - tf.reduce_mean(fields["AveBedrms"]))/tf.math.reduce_std(fields["AveBedrms"])
        std_population = (fields["Population"] - tf.reduce_mean(fields["Population"]))/tf.math.reduce_std(fields["Population"])
        std_households = (fields["AveOccup"] - tf.reduce_mean(fields["AveOccup"]))/tf.math.reduce_std(fields["AveOccup"])


        import tensorflow_transform as tft

        X = tf.stack([
        std_housing_median_age,
        std_total_rooms,
        std_total_bedrooms,
        std_households,
        std_population,
        std_median_income], axis=1)

        return X, fields["target"]


    def preprocess(self, serialized_record):
        import tensorflow as tf

        feature_description = {
        "MedInc": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "HouseAge": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "AveRooms": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "AveBedrms": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "Population": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "AveOccup": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "Latitude": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "Longitude": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "target": tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
        }

        # feature_description = {
        # "longitude": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "latitude": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "housing_median_age": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "total_rooms": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "total_bedrooms": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "population": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "households": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "median_income": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "median_house_value": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        # "ocean_proximity": tf.io.FixedLenFeature([], tf.string, default_value="")
        # }
        fields = tf.io.parse_example(serialized_record,feature_description)
        import tensorflow_transform as tft
        
        # std_longitude = tft.scale_to_z_score(fields["longitude"] - tft.mean(fields["longitude"]))
        # std_latitude = tft.scale_to_z_score(fields["latitude"] - tft.mean(fields["latitude"]))
        # std_housing_median_age = (fields["housing_median_age"] - tf.reduce_mean(fields["housing_median_age"]))/tf.math.reduce_std(fields["housing_median_age"])
        # std_total_rooms = (fields["total_rooms"] - tf.reduce_mean(fields["total_rooms"]))/tf.math.reduce_std(fields["total_rooms"])
        # std_total_bedrooms = (fields["total_bedrooms"] - tf.reduce_mean(fields["total_bedrooms"]))/tf.math.reduce_std(fields["total_bedrooms"])
        # std_population = (fields["population"] - tf.reduce_mean(fields["population"]))/tf.math.reduce_std(fields["population"])
        # std_households = (fields["households"] - tf.reduce_mean(fields["households"]))/tf.math.reduce_std(fields["households"])
        # std_median_income = (fields["median_income"] - tf.reduce_mean(fields["median_income"]))/tf.math.reduce_std(fields["median_income"])
        # ocean_proximity_id = tft.compute_and_apply_vocabulary(fields["ocean_proximity"])
        
        # X = tf.stack([
        #             # std_longitude,
        #             # std_latitude,
        #             fields["housing_median_age"],
        #             fields["total_rooms"],
        #             fields["total_bedrooms"],
        #             fields["population"],
        #             fields["households"],
        #             fields["median_income"]], axis=1)

        X = tf.stack([
            # std_longitude,
            # std_latitude,
            fields["HouseAge"],
            fields["AveRooms"],
            fields["AveBedrms"],
            fields["AveOccup"],
            fields["Population"],
            fields["MedInc"]], axis=1)

        # X = tf.stack([
        #             # std_longitude,
        #             # std_latitude,
        #             std_housing_median_age,
        #             std_total_rooms,
        #             std_total_bedrooms,
        #             std_population,
        #             std_households,
        #             std_median_income])
        # X = tf.transpose(X)
        tf.print(X[0])
        tf.print(fields["target"])
        # return X, fields["median_house_value"]
        return X, fields["target"]

    def create_housing_proto_buf_dataset(self, filepaths, repeat=None, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):
        import tensorflow as tf
        dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
        dataset = dataset.interleave(
            lambda filepath:  tf.data.TFRecordDataset(filepath).skip(1),
            cycle_length=n_readers, num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.preprocess_pikle_set, num_parallel_calls=n_parse_threads)
        return dataset.prefetch(1)

    #endregion

    #endregion

#endregion



















    def Build_Keras_Model(self):
        import tensorflow as tf
        from tensorflow import keras

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(30, name="First_HLayer", activation="relu", input_shape=[6]))
        model.add(keras.layers.Dense(30, name="Second_HLayer", activation="relu"))
        model.add(keras.layers.Dense(1, name="Output_HLayer"))

        model.compile(loss="mean_squared_error",
        optimizer="adam", metrics=["MeanSquaredError"])

        return model


    def fit_model(self, train, valid, checkpoint_path=None, model_param_path=None, no_epochs=30, batch_size=32, initial_epoch = 0, verbosity=1, get_model = False):
        
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
    
        callbacks  = callbacks + [lr_decay_callback, ckpt_callback]

        if get_model:
            return model

        # Start/resume training
        model.fit(train,
        # batch_size=batch_size,
        epochs=no_epochs,
        initial_epoch=initial_epoch,
        verbose=verbosity,
        # validation_steps=batch_size,
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
        print(f"Predicted value : {model.predict(test_x)}")



housing_model = california_housing() 
housing_model.create_tf_proto_buf_files()
train_files, test_files, validation_files = housing_model.get_tf_list_file_dataset()
train_set = housing_model.create_housing_proto_buf_dataset(train_files, repeat=1)
test_set = housing_model.create_housing_proto_buf_dataset(test_files, repeat=1)
validation_set = housing_model.create_housing_proto_buf_dataset(validation_files, repeat=1)


from CommonConstants import MODEL_PATH_CONSTANTS


#region Check point with variable learning rate

Total_Epochs = 20
Batch_Size = 32 #"Friends don’t let friends use mini-batches larger than 32“
Initiate_at_Epoch = 10
Initiate_at_Loss = 0.51

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)

# housing_model.fit_model(train_set , validation_set, None, None, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)

housing_model.model_predict(CheckPointPath, test_set)

# print(f"Actual value : {test_y}")
#endregion

#region hyper param explorer with scikit learn

# Total_Epochs = 2
# Batch_Size = 32 #"Friends don’t let friends use mini-batches larger than 32“
# Initiate_at_Epoch = 0
# Initiate_at_Loss = 0
# housing_model.Explore_Hyper_Params(train_x , train_y, validation_x, validation_y, None, None, Total_Epochs, Batch_Size, Initiate_at_Epoch, 1)

#endregion