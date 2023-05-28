import sys, os

from pathlib import Path 


import matplotlib.pyplot as plt



from CommonConstants import DATASET_PATH_CONSTANTS

class PlotData():

    def PlotHistogram(self, _panda_data_frame, _bins, _figsize, _title = "Histogram"):
        None

    def PlotColumnHistogram(self, _panda_data_frame, _columnID):
        _panda_data_frame[_columnID].hist()

    def Plot2DScatter(self, _panda_data_frame, _columnIdX, _columnIdY, _alpha):
        None

      
if __name__ == '__main__':
    from pathlib import Path
    
    DATA_FILE_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME,  DATASET_PATH_CONSTANTS.BLOB)

    from LoadData import LoadData
    objLoadData = LoadData()
    pandas_frame = objLoadData.load_Data(DATA_FILE_PATH)
    

    objPlotData = PlotData()
    
    # Plot Dataset
    # objPlotData.PlotColumnHistogram(pandas_frame, "target")
    # plt.show()

    #Perform Stratified Split

    from Splitdata import SplitData
    objSplitData = SplitData()

    strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="target")
    

    #Plot Test, Train Sets
    objPlotData.PlotColumnHistogram(strat_train_set, 'target')
    objPlotData.PlotColumnHistogram(strat_test_set, 'target')

    plt.show()