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



class imdbModel :

#region DataProcessing

    #region To get datasets as tf dataset

    def get_tf_list_file_dataset(self):

        return [DATASET_PATH_CONSTANTS.GetTrainSplitPath().format("*")], \
            [DATASET_PATH_CONSTANTS.GetTestSplitPath().format("*")], \
                [DATASET_PATH_CONSTANTS.GetValidationSplitPath().format("*")]

    def get_tfds_datasets(self, Fetch_Tuple = False):
        from LoadData import LoadData_TFDS
        objLoadData = LoadData_TFDS()
        import tensorflow_datasets as tfds 
        train_split, valid_split, test_split  = ["train[:85%]", "train[85%:100%]", "test"]

        dataset_train = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), Fetch_Tuple, train_split)
        dataset_test = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), Fetch_Tuple, test_split)
        dataset_valid = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), Fetch_Tuple, valid_split)
       
        return dataset_train, dataset_test,dataset_valid

    #endregion



    #region preprocess data

    def preprocess(self, X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch


    def Build_Vocabulary(self, tfdatasets, vocab_size=10000, num_oov_buckets = 1000):
        import tensorflow as tf
        from collections import Counter
        self.vocabulary = Counter()
        for dataset in tfdatasets:
            for X_batch, Y_batch in dataset.batch(32).map(self.preprocess):
                for review in X_batch:
                    self.vocabulary.update(list(review.numpy()))

        truncated_vocabulary = [
            word for word, count in self.vocabulary.most_common()[:vocab_size]]

        vocab_words = [
             word.decode("utf-8") for word, count in self.vocabulary.most_common()[:vocab_size]]
        
        words = tf.constant(truncated_vocabulary)
        word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
        oov_ids = tf.range(len(truncated_vocabulary), len(truncated_vocabulary)+num_oov_buckets, dtype=tf.int64)

        log_dir=MODEL_PATH_CONSTANTS().GetLogsPath()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save Labels separately on a line-by-line manner.
        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            import csv
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(['id'], ['word']))
            writer.writerows(zip(range(len(truncated_vocabulary)), vocab_words))
            writer.writerows(zip(range(len(truncated_vocabulary), len(vocab_words)+num_oov_buckets), ["UNKLNOWN"]*num_oov_buckets))

        vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
        self.table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

    def encode_words(self, X_batch, y_batch):
        return self.table.lookup(X_batch), y_batch

    def create_tfds_train_dataset_streams(self, dataset, repeat=None, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):
        dataset = dataset.batch(batch_size).map(self.preprocess)
        dataset = dataset.map(self.encode_words).prefetch(1)
        return dataset

    #endregion


#endregion

    def do_mask(self, X):
        from tensorflow import keras
        K = keras.backend
        X = (lambda inputs: K.not_equal(X, 0))
        return X


    def Build_Model(self, vocab_size=10000, num_oov_buckets=1000, embed_size = 128):
        from tensorflow import keras
        inputs = keras.layers.Input(shape=[None])
        # mask = keras.layers.Lambda(self.do_mask)(inputs)
        z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, mask_zero=True)(inputs)
        z = keras.layers.GRU(128, return_sequences=True, dropout=0.2)(z)
        z = keras.layers.GRU(128, dropout=0.2)(z)
        outputs = keras.layers.Dense(1, activation="sigmoid")(z)
        model = keras.Model(inputs=[inputs], outputs=[outputs])
    
        model.compile(loss="binary_crossentropy", optimizer="adam",
        metrics=["accuracy"])
        return model



    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


    def fit_model(self, Build_Model, train, valid, checkpoint_path=None, model_param_path=None, no_epochs=30, initial_epoch = 0, verbosity=1, get_model = False):
        
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
            model = Build_Model()
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
        model = load_model(checkpoint_path)
        print(f"Predicted value : {model.predict(test_x)}")
 
    def save_embeddings(self, checkpoint_path):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from tensorboard.plugins import projector

        from tensorflow.keras.models import load_model
        model = load_model(checkpoint_path)


        log_dir=MODEL_PATH_CONSTANTS().GetLogsPath()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        # Save the weights we want to analyze as a variable. Note that the first
        # value represents any unknown word, which is not in the metadata, here
        # we will remove this value.
        print(f"Number of layers : {len(model.layers)}")
        weights = tf.Variable(model.layers[1].get_weights()[0][1:])
        # Create a checkpoint from embedding, the filename and key are the
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        # Set up config.
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, config)





imdbModel_class = imdbModel()

train, test, validation = imdbModel_class.get_tfds_datasets(Fetch_Tuple = True)
imdbModel_class.Build_Vocabulary([train, test, validation])
train_set = imdbModel_class.create_tfds_train_dataset_streams(train, repeat=1, batch_size=32)
test_set = imdbModel_class.create_tfds_train_dataset_streams(test, repeat=1, batch_size=1)
validation_set = imdbModel_class.create_tfds_train_dataset_streams(validation, repeat=1, batch_size=32)

from CommonConstants import MODEL_PATH_CONSTANTS


#region Check point with variable learning rate

Total_Epochs = 20
Initiate_at_Epoch = 4
Initiate_at_Loss = 0.45

CheckPointPath = MODEL_PATH_CONSTANTS.GetAbsoluteCheckPointPath(Initiate_at_Epoch, Initiate_at_Loss)
ModleParamtPath = MODEL_PATH_CONSTANTS.GetAbsoluteModelParamPath(Initiate_at_Epoch)

# imdbModel_class.fit_model(imdbModel_class.Build_Model, train_set , validation_set, None, None, Total_Epochs, Initiate_at_Epoch, 1)


# imdbModel_class.model_predict(CheckPointPath, test_set.take(1))
# for x, y in test_x:
#     print(f"Actual value : {y}")

# imdbModel_class.save_embeddings(CheckPointPath)

review, y = imdbModel_class.preprocess(["This is a bad movie"], None)
review, y = imdbModel_class.encode_words(review, None)

imdbModel_class.model_predict(CheckPointPath, review)

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