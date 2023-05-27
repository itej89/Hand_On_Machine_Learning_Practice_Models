import sys, os
import tarfile
import urllib.request

from pathlib import Path 





from IDownloadData import IDownloadData

class DownloadData(IDownloadData):
    fileName = "DataBlob"
    data_dir = "DataSets"
    
    def __init__(self, _data_dir="DataSets" ,_fileName="DataBlob"):
        self.fileName = _fileName
        self.data_dir = _data_dir

    def download_data(self, url):
        import numpy as np
        X = 2*np.random.rand(100, 1)
        y = 4+3*X+np.random.randn(100, 1)
        X = X.reshape(-1)
        y = y.reshape(-1)
        import pandas as pd
        data_frame = pd.DataFrame(list(zip(X,y)), columns=['X', 'y'])

        from pathlib import Path
        import os
        blobPath = os.path.join(self.data_dir, self.fileName)
        os.makedirs(os.path.dirname(blobPath), exist_ok=True)
        self.save_data(data_frame, blobPath)
        
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
    
    DATA_PATH = os.path.join(pwd.parent, "datasets", "linear_rand")

    objDownloadData = DownloadData(DATA_PATH, "linear_rand.pkl")
    objDownloadData.download_data(None)






