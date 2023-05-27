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

        return strat_train_set, strat_test_set


                
if __name__ == '__main__':
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    Data_File_Path = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME, DATASET_PATH_CONSTANTS.BLOB)
    
    from LoadData import LoadData
    objLoadData = LoadData()
    pandas_frame = objLoadData.load_Data(Data_File_Path)


    import numpy as np
    pandas_frame = objLoadData.categorize_value_column(pandas_frame, "MedInc", "income_category", \
    _bins=[0.,1.5,3.0,4.5,6.0,np.inf], _labels=[1,2,3,4,5])
   


    objSplitData = SplitData()
    # train_set, test_set = objSplitData.split_data(pandas_frame, _test_percent=0.2, _randome_state=42)
    # print(len(train_set))
    # print(len(test_set))

    strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")
    print(len(strat_train_set))
    print(len(strat_test_set))