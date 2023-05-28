import abc

class IDownloadData(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'download_data') and 
                callable(subclass.download_data) and
                hasattr(subclass, 'extract_tar') and
                callable(subclass.extract_tar) and
                NotImplemented)
    
    @abc.abstractclassmethod
    def download_data(self, url):
        pass

    @abc.abstractclassmethod
    def extract_tar_data_file(self):
        pass