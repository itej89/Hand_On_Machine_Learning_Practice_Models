import sys, os
from pathlib import Path

from tensorflow.python.ops.gen_array_ops import inplace_add 







from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class LoadData:

    def peek_Data(self, _filename):
        data = self.load_Data(_filename) 
        print(f"Sample Data : \n {data}")

    def type_of_Data(self, _filename):
        data = self.load_Data(_filename) 
        print(f"Type of data : {type(data)}")

    def load_csv_data(self, _filename):
        import numpy as np
        import pandas as pd
        data_frame = pd.read_csv(_filename)
        for column in data_frame:
            if column != "k_median_house_value":
                print("Column Type : {}".format(data_frame.dtypes[column]))
                if data_frame.dtypes[column] == np.float64:
                    data_frame[column].fillna(0.0, inplace=True)
                    from sklearn.preprocessing import StandardScaler
                    standars_scalar = StandardScaler()
                    np_array = standars_scalar.fit_transform(data_frame[[column]])
                    data_frame[column] = pd.Series(np_array.flatten())
        print(data_frame)
        data_frame.to_csv(DATASET_PATH_CONSTANTS.GetNormalizedCSVPath())
        return data_frame

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
        pandas_frame.to_csv(DATASET_PATH_CONSTANTS.GetPickleAsCSVPath())
        return pandas_frame

    #convert a value column to categories using value bins
    def categorize_value_column(self, _pandas_frame, _column_label, _catogorized_column, _bins, _labels):
        _pandas_frame[_catogorized_column] = pd.cut(_pandas_frame[_column_label], bins=_bins, labels=_labels)
        return _pandas_frame;
    
if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.CSV)


    objDownloadData = LoadData()
    housing = objDownloadData.load_csv_data(Data_File_Path)