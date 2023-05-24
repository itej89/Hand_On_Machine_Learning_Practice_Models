import sys, os
import tarfile
import urllib.request

from pathlib import Path 


class DownloadData():
    fielName = "DataBlob"
    data_dir = "DataSets"
    
    def __init__(self, _data_dir="DataSets" ,_fileName="DataBlob"):
        self.fileName = _fileName
        self.data_dir = _data_dir

    def download_data(self, url):
        os.makedirs(url, exist_ok=True)
        file_path= os.path.join(self.data_dir, self.fileName)
        print(f'Downloadiong Data from : {url}')
        urllib.request.urlretrieve(url, file_path)
        print(f'Downloaded Data to : {file_path}')
    
    def extract_tar_data_file(self):
        file_path= os.path.join(self.data_dir, self.fileName)
        tarDatafile = tarfile.open(file_path)
        print(f'Extracting : {file_path}')
        tarDatafile.extractall(path=self.data_dir)
        tarDatafile.close()
        print(f'Extracting finished.')

    
if __name__ == '__main__':
    #Download Data
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_URL = DOWNLOAD_ROOT+"datasets/housing/housing.tgz"
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    HOUSING_PATH = os.path.join(pwd.parent, "datasets", "housing")

    objDownloadData = DownloadData(HOUSING_PATH, "housing.tgz")
    objDownloadData.extract_tar_data_file()






