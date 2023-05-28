import sys, os

from pathlib import Path 







from CommonConstants import DATASET_PATH_CONSTANTS

import pandas as pd

class LoadData:


    def generate_time_Series(self, batch_size, n_steps):
        import numpy as np
        np.random.seed(42)
        freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
        time = np.linspace(0, 1, n_steps)
        # print(offset1)
        # print(time)
        # print(len(time - offset1))
        series = 0.5 * np.sin((time - offset1) * (freq1 * 10  +10))
        series += 0.2 * np.sin((time - offset2) * (freq2 * 20  +20))
        series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
        return series[..., np.newaxis].astype(np.float32)

    def load_generated_data(self, instances, feature_len, target_len):
        return self.generate_time_Series(instances, feature_len+target_len)

    def load_Data(self, _filename, _showinfo=False):
        pass

    def scikit_to_pandas(self, data_set):
        pass

    
if __name__ == '__main__':
    #Download Data
    from pathlib import Path
    
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)


    objDownloadData = LoadData()
    housing = objDownloadData.peek_Data(Data_File_Path)