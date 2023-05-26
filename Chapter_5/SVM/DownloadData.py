import sys, os

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent.parent, "CommonInterfaces"))


from IDownloadData import IDownloadData

from CommonConstants import DATASET_PATH_CONSTANTS

class DownloadData(IDownloadData):
    fileName = "DataBlob"
    data_dir = "DataSets"
    
    def __init__(self, _data_dir="DataSets" ,_fileName="DataBlob"):
        self.fileName = _fileName
        self.data_dir = _data_dir

    def download_data(self, url):
        from sklearn import datasets
        iris = datasets.load_iris()

        from pathlib import Path
        import os
        blobPath = os.path.join(self.data_dir, self.fileName)
        os.makedirs(os.path.dirname(blobPath), exist_ok=True)
        self.save_data(iris, blobPath)
        
    def extract_tar_data_file(self):
        None

    def save_data(self, data, filename):
        if os.path.exists(filename):
            os.remove(filename)

        file_object  = open(filename, "wb") 

        import pickle
        pickle.dump(data, file_object)





    
if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

    objDownloadData = DownloadData(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB)
    objDownloadData.download_data(None)








