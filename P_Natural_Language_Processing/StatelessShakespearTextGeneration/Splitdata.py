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

    def split_text(self, data_set, train_split, test_split, valid_split):
        data_len = len(data_set)
        import tensorflow as tf
        train_size = data_len * train_split // 100
        test_size = data_len * test_split // 100
        valid_size = data_len * valid_split // 100
        train = data_set[:train_size]
        test = data_set[train_size:(train_size+test_size)]
        valid = data_set[train_size+test_size:]

        return train , test, valid

                
if __name__ == '__main__':

    from LoadData import LoadData, Tokenizer
    objLoadData = LoadData()
    strData = objLoadData.load_Data(DATASET_PATH_CONSTANTS.GetBlobPath())

    tokenizer = Tokenizer()
    encoded_data = tokenizer.Tokenize(strData)


    split_data = SplitData()
    train, test, valid  = split_data.split_text(encoded_data, 75, 15, 10)

    print(len(encoded_data))
    print(len(train)+len(test)+len(valid))
    print(len(train))