import sys, os

from pathlib import Path 


import pandas as pd

class LoadData():

    def load_Data(self, _filename, _showinfo=False):
        file_object = open(_filename, "rb")
        import pickle
        pandas_data_frame = pickle.load(file_object)
        return pandas_data_frame
    
    def categorize_value_column(self, _pandas_frame, _column_label, _catogorized_column, _bins, _labels):
        _pandas_frame[_catogorized_column] = pd.cut(_pandas_frame[_column_label], bins=_bins, labels=_labels)
        return _pandas_frame;

if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    Data_File_Path = os.path.join(pwd.parent, "datasets", "linear_rand", "linear_rand.pkl")


    objDownloadData = LoadData()
    df = objDownloadData.load_Data(Data_File_Path, True)
    print(df)