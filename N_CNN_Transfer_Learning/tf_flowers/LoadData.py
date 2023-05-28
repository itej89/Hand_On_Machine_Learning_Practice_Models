from os.path import split
import sys, os

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))

from CommonConstants import DATASET_PATH_CONSTANTS


class LoadData_TFDS():

    def load_Data(self):
        pass

    def load_Data_TFDS(self, name, data_dir, tuple_format = True, Split=None):
        import tensorflow_datasets as tfds
        dataset = tfds.load(name=name, data_dir=data_dir, download=False, as_supervised = tuple_format, split =Split)
        return dataset
    
    def get_splits_TFDS(self, tfds_dataset):
        splits = []
        for key in tfds_dataset:
            splits.append(key)
    
    def get_splits_TFDS(self, tfds_dataset):
        splits = []
        for key in tfds_dataset:
            splits.append(key)

        return splits
    
    def get_dataset_dict_headers_TFDS(self, tfds_dataset_split):
        labels = []
        for x in tfds_dataset_split:
            for key in x:
                labels.append(key)
            break;

        return labels
    
if __name__ == '__main__':
    #Download Data
    objDownloadData = LoadData_TFDS()
    dataset = objDownloadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), False)
    splits = objDownloadData.get_splits_TFDS(dataset)                                                                                                
    labels = objDownloadData.get_dataset_dict_headers_TFDS(dataset[splits[0]].batch(1))
    print(splits)
    print(labels)
