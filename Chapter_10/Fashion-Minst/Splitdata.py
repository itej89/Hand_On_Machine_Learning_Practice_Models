import sys, os

from pathlib import Path 




import matplotlib.pyplot as plt



from CommonConstants import DATASET_PATH_CONSTANTS

class SplitData:

    def split_data(self, data_Set, _test_percent, _randome_state):
        None

    def stratified_split_data(self, _panda_data_frame, _n_splits , _test_percent, _randome_state, _columnID):
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=_n_splits, test_size=_test_percent, random_state=_randome_state)
        
        import pandas as pd
        strat_train_set = pd.DataFrame(data=None, columns=_panda_data_frame.columns)
        strat_test_set = pd.DataFrame(data=None, columns=_panda_data_frame.columns)

        import numpy as np
        for train_index, test_index in split.split(_panda_data_frame, _panda_data_frame[_columnID]):
            strat_train_set = pd.concat([strat_train_set, _panda_data_frame.loc[train_index]], ignore_index=True)
            strat_test_set = pd.concat([strat_test_set, _panda_data_frame.loc[test_index]], ignore_index=True)

        return strat_train_set, strat_test_set

    def split_train_to_validation(self, X_train_full, y_train_full):
        X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
        return X_train, y_train, X_valid, y_valid

                
if __name__ == '__main__':
    from pathlib import Path
    
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)
    
    from LoadData import LoadData
    objLoadData = LoadData()
    X_train, y_train, X_test, y_test = objLoadData.load_Data(Data_File_Path)




    objSplitData = SplitData()

    X_train, y_train, X_valid, y_valid = objSplitData.split_train_to_validation(X_train, y_train)
    print(len(X_train))
    print(len(X_valid))