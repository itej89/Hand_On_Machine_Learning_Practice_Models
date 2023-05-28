import sys, os

from pathlib import Path 

from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class LoadData:

    def load_Data(self, _filename, _showinfo=False):
        file_object = open(_filename, "rb")
        import pickle
        (x_train, y_train), (X_test, y_test) = pickle.load(file_object) 
        return x_train, y_train, X_test, y_test

    def get_class_names():
        return ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)


    objDownloadData = LoadData()
    x_train, y_train, X_test, y_test = objDownloadData.load_Data(Data_File_Path, True)