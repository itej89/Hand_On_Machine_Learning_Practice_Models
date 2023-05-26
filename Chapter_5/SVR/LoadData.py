import sys, os

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent.parent, "CommonInterfaces"))


from ILoadData import ILoadData

from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class LoadData(ILoadData):

    def peek_Data(self, _filename):
        data = self.load_Data(_filename) 
        print(f"Sample Data : \n {data}")

    def type_of_Data(self, _filename):
        data = self.load_Data(_filename) 
        print(f"Type of data : {type(data)}")

    def load_Data(self, _filename, _showinfo=False):
        file_object = open(_filename, "rb")
        import pickle
        data = pickle.load(file_object) 
        return self.scikit_to_pandas(data)

    def scikit_to_pandas(self, data_set):
        import pandas as pd
        import numpy as np
        pandas_frame = pd.DataFrame(data= np.c_[data_set['data'], data_set['target']],
                            columns= data_set['feature_names'] + ['target'])
        return pandas_frame

    #convert a value column to categories using value bins
    def categorize_value_column(self, _pandas_frame, _column_label, _catogorized_column, _bins, _labels):
        _pandas_frame[_catogorized_column] = pd.cut(_pandas_frame[_column_label], bins=_bins, labels=_labels)
        return _pandas_frame;
    
if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)


    objDownloadData = LoadData()
    housing = objDownloadData.peek_Data(Data_File_Path)