import sys, os

from pathlib import Path 




import matplotlib.pyplot as plt



from CommonConstants import DATASET_PATH_CONSTANTS

class SplitData:

    def split_data(self, data_Set, _test_percent, _randome_state):
        None

    def stratified_split_data(self, _panda_data_frame, _n_splits , _test_percent, _randome_state, _columnID):
        pass

    def split_tfds(self, data_set, splits):
        return (lambda x : data_set[s] for s in splits)

                
if __name__ == '__main__':

    from LoadData import LoadData_TFDS
    objLoadData = LoadData_TFDS()
    dataset_train = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), True, "train")
    dataset_test = objLoadData.load_Data_TFDS(DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.GetDirectoryPath(), True, "test")
    dataset_train = dataset_train.batch(1).prefetch(1)
    for x,y in dataset_train:
        print(x, y)
        break