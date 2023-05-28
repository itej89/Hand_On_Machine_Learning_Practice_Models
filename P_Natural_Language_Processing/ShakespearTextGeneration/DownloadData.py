import sys, os
from pathlib import Path 




from tensorflow import keras

from IDownloadData import IDownloadData
from CommonConstants import DATASET_PATH_CONSTANTS


class DownloadData(IDownloadData):
    fileName = "DataSetName"
    data_dir = "DataSets"
    
    def __init__(self, _data_dir="DataSets" ,_fileName="DataSetName"):
        self.fileName = _fileName
        self.data_dir = _data_dir

    def download_data(self, url):
        pass

    def download_data(self, url, filename, blobpath):
        shakespeare_url = url # shortcut URL
        filepath = keras.utils.get_file(filename, shakespeare_url)
        with open(filepath) as f:
            data = f.read()
            self.save_data(data, blobpath)


    

    def download_data_TFDS(self, name, data_dir):
        import tensorflow_datasets as tfds
        tfds.load(name=name, data_dir=data_dir, download=True)

        
    def extract_tar_data_file(self):
        pass

    def save_data(self, data, filename):
        if os.path.exists(filename):
            os.remove(filename)
        else:
            DATASET_PATH_CONSTANTS.CreatePath(DATASET_PATH_CONSTANTS.GetDirectoryPath())
            
        file_object  = open(filename, "wb") 

        import pickle
        pickle.dump(data, file_object)

    
if __name__ == '__main__':
    #Download Data

    objDownloadData = DownloadData()
    objDownloadData.download_data(DATASET_PATH_CONSTANTS.URL_FILE.URL, DATASET_PATH_CONSTANTS.URL_FILE.FILE, DATASET_PATH_CONSTANTS.GetBlobPath())








