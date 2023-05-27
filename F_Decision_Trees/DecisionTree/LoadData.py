import sys, os

from pathlib import Path 



from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class LoadData():

    def __init__(self) -> None:
        self.feature_names = None
        self.target_names = None
        super().__init__()


    def get_feature_names(self):
        return self.feature_names

    def get_target_names(self):
        return self.target_names

    def load_Data(self, _filename, _showinfo=False):
        file_object = open(_filename, "rb")
        import pickle
        iris = pickle.load(file_object) 
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names.tolist()
        return self.scikit_to_pandas(iris)

    def scikit_to_pandas(self, data_set):
        import pandas as pd
        import numpy as np
        pandas_frame = pd.DataFrame(data= np.c_[data_set['data'], data_set['target']],
                            columns= data_set['feature_names'] + ['target'])
        return pandas_frame

if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)


    objDownloadData = LoadData()
    df = objDownloadData.load_Data(Data_File_Path, True)
    print(df)