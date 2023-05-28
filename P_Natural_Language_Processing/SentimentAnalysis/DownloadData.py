import sys, os

from pathlib import Path 


from CommonConstants import DATASET_PATH_CONSTANTS

class DownloadData_TFDS():
    fileName = "DataSetName"
    data_dir = "DataSets"
    
    def __init__(self, _data_dir="DataSets" ,_fileName="DataSetName"):
        self.fileName = _fileName
        self.data_dir = _data_dir

    def download_data(self, url):
        pass

    def download_data_TFDS(self, name, data_dir):
        import tensorflow_datasets as tfds
        tfds.load(name=name, data_dir=data_dir, download=True)

        
    def extract_tar_data_file(self):
        pass

    def save_data(self, data, filename):
        pass

    
if __name__ == '__main__':
    #Download Data

    objDownloadData = DownloadData_TFDS()
    objDownloadData.download_data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath())








