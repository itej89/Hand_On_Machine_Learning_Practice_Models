import sys, os

from pathlib import Path

from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.text import Tokenizer 







from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class Tokenizer:
    tokenizer = None
    max_id = 0
    def Tokenize(self, text_data):
        import numpy as np
        from tensorflow import keras
        self.tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts([text_data])
        self.max_id = len(self.tokenizer.word_index)
        [encoded] =  np.array(self.tokenizer.texts_to_sequences([text_data]))
        return encoded

class LoadData:

    def load_Data(self, _filename, _showinfo=False):
        file_object = open(_filename, "rb")
        import pickle
        return pickle.load(file_object)


    def scikit_to_pandas(self, data_set):
        pass

    
if __name__ == '__main__':

    objDownloadData = LoadData()
    strData = objDownloadData.load_Data(DATASET_PATH_CONSTANTS.GetBlobPath())

    tokenizer = Tokenizer()
    encoded_data = tokenizer.Tokenize(strData)
    print(len(encoded_data))