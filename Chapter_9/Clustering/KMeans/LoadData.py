import sys, os

from pathlib import Path 







from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class LoadData:

    def load_Data(self, _filename, _showinfo=False):
        file_object = open(_filename, "rb")
        import pickle
        iris = pickle.load(file_object) 
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
    pwd = Path(os.path.abspath(__file__))
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)


    objDownloadData = LoadData()
    df = objDownloadData.load_Data(Data_File_Path, True)
    print(df)