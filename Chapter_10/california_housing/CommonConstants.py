class DATASET_PATH_CONSTANTS:
    DIR = "datasets"
    NAME = "california_housing"
    BLOB = "california_housing.pkl"

    @staticmethod
    def GetRootDirectory():
        import os
        from pathlib import Path
        return Path(os.path.abspath(__file__)).parent

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
    MODEL_EXTENSION = "WideDeep"
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