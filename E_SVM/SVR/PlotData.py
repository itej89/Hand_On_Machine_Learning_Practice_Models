import sys, os

from pathlib import Path 




from CommonConstants import DATASET_PATH_CONSTANTS

import matplotlib.pyplot as plt



class PlotData:

    def PlotHistogram(self, _panda_data_frame, _bins, _figsize):
        _panda_data_frame.hist(bins=_bins, figsize=_figsize)
        plt.show()

    def PlotColumnHistogram(self, _panda_data_frame, _columnID):
        _panda_data_frame[_columnID].hist()

    def Plot2DScatter(self, _panda_data_frame, _columnIdX, _columnIdY, _alpha):
        _panda_data_frame.plot(kind="scatter", x=_columnIdX, y=_columnIdY, alpha=_alpha)
        plt.show()

    def Plot2DScatterExtra(self, _panda_data_frame, _columnIdX, _columnIdY, _alpha, 
    _radius_columnID, _label, _figsize, _color_column_ID):
        _panda_data_frame.plot(kind="scatter", x=_columnIdX, y=_columnIdY, alpha=_alpha,
        s=_panda_data_frame[_radius_columnID]/100, label=_label, figsize=_figsize,
        c=_color_column_ID, cmap=plt.get_cmap("jet"), colorbar=True)
        plt.legend()
        plt.show()

    def PlotCorrelationMatrix(self, _panda_data_frame, _attributes, _figsize):
        from pandas.plotting import scatter_matrix
        scatter_matrix(_panda_data_frame[_attributes], figsize = _figsize)
        plt.show()


if __name__ == '__main__':
    from pathlib import Path
    
    DATA_FILE_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME,  DATASET_PATH_CONSTANTS.BLOB)

    from LoadData import LoadData
    objLoadData = LoadData()
    pandas_frame=  objLoadData.load_Data(DATA_FILE_PATH)
    

    objPlotData = PlotData()
    # objPlotData.PlotHistogram(pandas_frame, 50, (20,15))


    import numpy as np
    pandas_frame = objLoadData.categorize_value_column(pandas_frame, "MedInc", "income_category", \
    _bins=[0.,1.5,3.0,4.5,6.0,np.inf], _labels=[1,2,3,4,5])
    # objPlotData.PlotColumnHistogram(pandas_frame, "income_category")


    # objPlotData.Plot2DScatterExtra(pandas_frame, "Longitude", "Latitude", 0.4, 
    # "Population", "Population", (10,7), "MedInc")


    # objPlotData.PlotCorrelationMatrix(pandas_frame, ["target", "MedInc", "AveRooms","HouseAge"], (12,8))
    
    # objPlotData.Plot2DScatter(pandas_frame, "MedInc", "target", 1.0)


    from Splitdata import SplitData
    objSplitData = SplitData()

    strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")
    print(len(strat_train_set))
    print(len(strat_test_set))
    objPlotData.PlotColumnHistogram(strat_train_set, 'target')
    objPlotData.PlotColumnHistogram(strat_test_set, 'target')

    plt.show()