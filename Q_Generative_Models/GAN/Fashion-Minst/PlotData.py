import sys, os

from pathlib import Path 




import matplotlib.pyplot as plt



from CommonConstants import DATASET_PATH_CONSTANTS

class PlotData:

    def PlotHistogram(self, _panda_data_frame, _bins, _figsize, _title = "Histogram"):
        None

    def PlotColumnHistogram(self, _panda_data_frame, _columnID):
        _panda_data_frame[_columnID].hist()

    def Plot2DScatter(self, _panda_data_frame, _columnIdX, _columnIdY, _alpha):
        None

    def numpy_print_all_unique(self, arr):
        import numpy as np
        print(np.unique(y_train, return_counts=False))


    def plot_image(self, image):
        plt.imshow(image, cmap="binary")
        plt.axis("off")

    def show_reconstructions(self, reconstructions, test_x, n_images):
        fig = plt.figure(figsize=(n_images * 1.5, 3))
        for image_index in range(n_images):
            plt.subplot(2, n_images, 1 + image_index)
            self.plot_image(test_x[image_index])
            plt.subplot(2, n_images, 1 + n_images + image_index)
            self.plot_image(reconstructions[image_index])
        
        plt.show()

    def plot_scatter(self, test_compressed, test_y):
        plt.scatter(test_compressed[:, 0], test_compressed[:, 1], c=test_y, s=10,cmap="tab10")
        plt.show()


    def shaow_images(self, test_compressed, n_images):
        fig = plt.figure(figsize=(n_images * 1.5, 3))
        for image_index in range(n_images):
            plt.subplot(2, n_images, 1 + image_index)
            self.plot_image(test_compressed[image_index])
        plt.show()

if __name__ == '__main__':
    from pathlib import Path
    pwd = Path(os.path.abspath(__file__))
    DATA_FILE_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, 
    DATASET_PATH_CONSTANTS.NAME,  DATASET_PATH_CONSTANTS.BLOB)

    from LoadData import LoadData
    objLoadData = LoadData()
    x_train, y_train, X_test, y_test =  objLoadData.load_Data(DATA_FILE_PATH)
    

    objPlotData = PlotData()

    import pandas as pd
    pandas_frame = pd.DataFrame(x_train, y_train, columns = ['input', 'target'])

    # Plot Dataset
    # objPlotData.PlotColumnHistogram(pandas_frame, "target")
    # plt.show()

    # #Perform Stratified Split

    from Splitdata import SplitData
    objSplitData = SplitData()

    strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="target")
    

    #Plot Test, Train Sets
    objPlotData.PlotColumnHistogram(strat_train_set, 'target')
    objPlotData.PlotColumnHistogram(strat_test_set, 'target')

    plt.show()