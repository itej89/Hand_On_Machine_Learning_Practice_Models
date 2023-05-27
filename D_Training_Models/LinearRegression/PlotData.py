import sys, os

from pathlib import Path 


import matplotlib.pyplot as plt

class PlotData():

    def PlotHistogram(self, _panda_data_frame, _bins, _figsize, _title = "Histogram"):
        _panda_data_frame.hist(bins=_bins)

    def PlotColumnHistogram(self, _panda_data_frame, _columnID):
        _panda_data_frame[_columnID].hist()

    def Plot2DScatter(self, _panda_data_frame, _columnIdX, _columnIdY, _alpha):
        _panda_data_frame.plot(title = "X, y Scatter", kind="scatter", x=_columnIdX, y=_columnIdY, alpha=_alpha)

                

if __name__ == '__main__':
    from pathlib import Path
    
    DATA_FILE_PATH = os.path.join(pwd.parent, "datasets", "linear_rand",  "linear_rand.pkl")

    from LoadData import LoadData
    objLoadData = LoadData()
    pandas_frame = objLoadData.load_Data(DATA_FILE_PATH)
    print(f"X max  : {pandas_frame['X'].max()}, y max : {pandas_frame['y'].max()}")

    objPlotData = PlotData()
    
    # Plot Dataset
    # objPlotData.Plot2DScatter(pandas_frame, 'X', 'y', 1.0)
    objPlotData.PlotHistogram(pandas_frame, 10, (10,5))

    #Perform Stratified Split
    import numpy as np
    pandas_frame = objLoadData.categorize_value_column(pandas_frame, "y", "y_cat", \
    _bins=[0.,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,np.inf], _labels=[1,2,3,4,5,6,7,8,9,10])
    print(pandas_frame)


    from Splitdata import SplitData
    objSplitData = SplitData()

    strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="y_cat")
    
    #Plot Test, Train Sets
    objPlotData.PlotColumnHistogram(strat_train_set, 'y')
    objPlotData.PlotColumnHistogram(strat_test_set, 'y')

    plt.show()

