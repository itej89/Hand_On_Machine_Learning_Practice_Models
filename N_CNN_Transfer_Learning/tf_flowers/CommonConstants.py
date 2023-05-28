class LABELS_TFDS:
    TARGET = "label"
    FEATURES = ["image"]

class LABELS_PKL:
    TARGET = ""
    FEATURES = []

class LABELS_CSV:
    TARGET = ""
    FEATURES = []

class DATASET_PATH_CONSTANTS:
    DIR = "datasets"
    NAME = "tf_flowers"
    BLOB = "tf_flowers.pkl"
    CSV = ""
    PKL_CSV = ""
    NORM_CSV = ""
    SPLIT_DIR = "split_csv"
    SPLIT_DIR_TRAIN = "train"
    SPLIT_DIR_VAL = "validation"
    SPLIT_DIR_TEST = "test"
    TRAIN_EXT = "_train_"
    VAL_EXT = "_val_"
    TEST_EXT = "_test_"
    DATA_EXT = ".tfrecord"

    @staticmethod
    def GetRootDirectory():
        import os
        from pathlib import Path
        return Path(os.path.abspath(__file__)).parent

    @staticmethod
    def GetCSVPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.CSV)
    
    @staticmethod
    def GetPickleAsCSVPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.PKL_CSV)
    
    @staticmethod
    def GetNormalizedCSVPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.NORM_CSV)
    
    @staticmethod
    def RemovePath(Dir):
        import os
        if os.path.exists(Dir):
            import shutil
            shutil.rmtree(Dir)


    @staticmethod
    def GetSplitDir():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR)

    @staticmethod
    def GetTrainSplitDir():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR, DATASET_PATH_CONSTANTS.SPLIT_DIR_TRAIN)

    staticmethod
    def GetTestSplitDir():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR, DATASET_PATH_CONSTANTS.SPLIT_DIR_TEST)

    @staticmethod
    def GetValidationSplitDir():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR, DATASET_PATH_CONSTANTS.SPLIT_DIR_VAL)



    @staticmethod
    def GetTrainSplitPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR, DATASET_PATH_CONSTANTS.SPLIT_DIR_TRAIN, DATASET_PATH_CONSTANTS.NAME+DATASET_PATH_CONSTANTS.TRAIN_EXT+"{}"+DATASET_PATH_CONSTANTS.DATA_EXT)


    @staticmethod
    def GetTestSplitPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR, DATASET_PATH_CONSTANTS.SPLIT_DIR_TEST, DATASET_PATH_CONSTANTS.NAME+DATASET_PATH_CONSTANTS.TEST_EXT+"{}"+DATASET_PATH_CONSTANTS.DATA_EXT)


    @staticmethod
    def GetValidationSplitPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.SPLIT_DIR, DATASET_PATH_CONSTANTS.SPLIT_DIR_VAL, DATASET_PATH_CONSTANTS.NAME+DATASET_PATH_CONSTANTS.VAL_EXT+"{}"+DATASET_PATH_CONSTANTS.DATA_EXT)


    @staticmethod
    def GetDirectoryPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

    @staticmethod
    def GetBlobPath():
        import os
        return os.path.join(DATASET_PATH_CONSTANTS.GetRootDirectory(), DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)

    @staticmethod
    def CreatePath(DirName):
        import os
        return os.makedirs(DirName, exist_ok=True)
        
class MODEL_PATH_CONSTANTS:
    DIR = "checkpoints"
    MODEL_EXTENSION = ""
    NAME = DATASET_PATH_CONSTANTS.NAME+MODEL_EXTENSION
    BLOB = DATASET_PATH_CONSTANTS.NAME+MODEL_EXTENSION+".h5"
    CHECKPOINT = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    LEARNING_RATE_DECAY_CALLBACK = "LRcallback.{epoch:02d}.pickle"
    MODEL_PARAM = "model-param.{epoch:02d}.pickle"
    VIZ_MODEL = DATASET_PATH_CONSTANTS.NAME+MODEL_EXTENSION+".png"
    LOGS = "logs"

    @staticmethod
    def GetRootDirectory():
        import os
        from pathlib import Path
        return Path(os.path.abspath(__file__)).parent

    @staticmethod
    def GetDirectoryPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME)
  
    @staticmethod
    def GetCheckPointPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, MODEL_PATH_CONSTANTS.CHECKPOINT)

    def GetAbsoluteCheckPointPath(epoch, val_loss):
        file_name = f"weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, file_name)


    @staticmethod
    def GetLRDecayCallbackPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, MODEL_PATH_CONSTANTS.LEARNING_RATE_DECAY_CALLBACK)

    @staticmethod
    def GetModelParamPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, MODEL_PATH_CONSTANTS.MODEL_PARAM)

    def GetAbsoluteModelParamPath(epoch):
        file_name = f"model-param.{epoch:02d}.pickle"
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, file_name)


    @staticmethod
    def GetBlobPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, MODEL_PATH_CONSTANTS.BLOB)

    @staticmethod
    def GetModelVizPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, MODEL_PATH_CONSTANTS.VIZ_MODEL)

    @staticmethod
    def GetLogsPath():
        import os
        return os.path.join(MODEL_PATH_CONSTANTS.GetRootDirectory(), MODEL_PATH_CONSTANTS.DIR, MODEL_PATH_CONSTANTS.NAME, MODEL_PATH_CONSTANTS.LOGS)

    @staticmethod
    def CreatePath(DirName):
        import os
        return os.makedirs(DirName, exist_ok=True)



# class LABELS:
#     STRAT_SPLIT_CAT_COLUMN = "target"
#     OUTPUT_COLUMN = "target"