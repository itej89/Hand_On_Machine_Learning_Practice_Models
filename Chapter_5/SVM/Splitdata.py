import sys, os

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent.parent, "CommonInterfaces"))

import matplotlib.pyplot as plt

from ISplitData import ISplitData

from CommonConstants import DATASET_PATH_CONSTANTS

class SplitData(ISplitData):

    def split_data(self, data_Set, _test_percent, _randome_state):
        from sklearn.model_selection import train_test_split
        return train_test_split(data_Set, test_size=_test_percent, random_state=_randome_state)

    def stratified_split_data(self, _panda_data_frame, _n_splits , _test_percent, _randome_state, _columnID):
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=_n_splits, test_size=_test_percent, random_state=_randome_state)
        
        import pandas as pd
        strat_train_set = pd.DataFrame(data=None, columns=_panda_data_frame.columns)
        strat_test_set = pd.DataFrame(data=None, columns=_panda_data_frame.columns)

        import numpy as np
        for train_index, test_index in split.split(_panda_data_frame, _panda_data_frame[_columnID]):
            strat_train_set = strat_train_set.append(_panda_data_frame.loc[train_index])  
            strat_test_set = strat_test_set.append(_panda_data_frame.loc[test_index])

        return strat_train_set, strat_test_set

        
if __name__ == '__main__':
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)
    
    from LoadData import LoadData
    objLoadData = LoadData()
    pandas_frame = objLoadData.load_Data(Data_File_Path)


    objSplitData = SplitData()

    strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="target")
    print(len(strat_train_set))
    print(len(strat_test_set))