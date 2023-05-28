import os


from CommonConstants import PATH_CONSTANTS, LABELS

class LogisticRegression_Iris :



    def split_data(self, pandas_frame):
        from Splitdata import SplitData
        objSplitData = SplitData()

        strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID=LABELS.STRAT_SPLIT_CAT_COLUMN)   

        if LABELS.STRAT_SPLIT_CAT_COLUMN != LABELS.OUTPUT_COLUMN:
            strat_train_set = strat_train_set.drop(LABELS.STRAT_SPLIT_CAT_COLUMN, axis=1)
            strat_test_set = strat_test_set.drop(LABELS.STRAT_SPLIT_CAT_COLUMN, axis=1)

        strat_train_set_data = strat_train_set.drop(LABELS.OUTPUT_COLUMN, axis=1)
        strat_train_set_labels = strat_train_set[LABELS.OUTPUT_COLUMN]

        strat_test_set_data = strat_test_set.drop(LABELS.OUTPUT_COLUMN, axis=1)
        strat_test_set_labels = strat_test_set[LABELS.OUTPUT_COLUMN]

        return strat_train_set_data, strat_train_set_labels, strat_test_set_data, strat_test_set_labels

    def get_data_sets(self):
        from pathlib import Path
        
        DATA_PATH = os.path.join(pwd.parent, PATH_CONSTANTS.DATASET_DIR, PATH_CONSTANTS.DATASET_NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data_frame = objLoadData.load_Data(os.path.join(DATA_PATH, PATH_CONSTANTS.DATASET_BLOB), False)

        # import numpy as np
        # data_frame = objLoadData.categorize_value_column(data_frame, "y", "y_cat", \
        # _bins=[0.,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,np.inf], _labels=[1,2,3,4,5,6,7,8,9,10])

        train_x , train_y, text_x, test_y = self.split_data(data_frame)
        
        return train_x , train_y, text_x, test_y

    def create_pipeline_verginica(self, data_frame):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from Verginica_Label_Filter_Transformer import Verginica_Label_Filter_Transformer
        
        num_pipeline = Pipeline([
            ('Verginica_Label_Filter_Transformer', Verginica_Label_Filter_Transformer())
        ])
        
        return num_pipeline

    def create_pipeline_petal_width(self, data_frame):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from Petal_Width_Filter_Transformer import Petal_Width_Filter_Transformer
        
        num_pipeline = Pipeline([
            ('Petal_Width_Filter_Transformer', Petal_Width_Filter_Transformer())
        ])

        return num_pipeline


    def create_pipeline_petal(self, data_frame):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from Petal_Filter_Transformer import Petal_Filter_Transformer
        
        num_pipeline = Pipeline([
            ('Petal_Filter_Transformer', Petal_Filter_Transformer())
        ])

        return num_pipeline
    
    def runthrough_pipeline(self, full_pipeline, data_frame):
        data_formated = full_pipeline.fit_transform(data_frame)
        return data_formated


    def get_num_attributes(self, data_frame):
        num_attribs = list(data_frame)
        return num_attribs

    def get_all_features(self, full_pipeline, data_frame):
        num_attribs = self.get_num_attributes(data_frame)
        return num_attribs

    def Perform_VerginicaDetection(self, X, y):
        from sklearn.linear_model import LogisticRegression
        log_Reg = LogisticRegression()
        log_Reg.fit(X,y)
        return log_Reg

    def Predict_VerginicaDetection(self, model, X_test, y_test):
        import numpy as np

        X_test = np.linspace(0, 3, 1000).reshape(-1, 1)
        y_proba = model.predict_proba(X_test)

        import matplotlib.pyplot as plt
        plt.plot(X_test, y_proba[:, 1], "g-", label="Iris-Virginica")
        plt.plot(X_test, y_proba[:, 0], "b--", label="Not Iris-Virginica")
        plt.show()



    def Predict_VerginicaPetalBasedDetection(self, model, X_test, y_test):
        import numpy as np

        limit_3cm = np.where(X_test[:,1] <= 3)[0]
        X_test = X_test[limit_3cm]

        y_proba = model.predict_proba(X_test)

        varginica_indices = np.where(y_proba[:,0] > 0.5)[0]
        X_test_verginica = X_test[varginica_indices]
        X_test_verginica = X_test_verginica.T
        
        non_varginica_indices = np.where(y_proba[:,0] < 0.5)[0]
        X_test_non_verginica = X_test[non_varginica_indices]
        X_test_non_verginica = X_test_non_verginica.T

        import matplotlib.pyplot as plt
        plt.scatter(x=X_test_verginica[0], y=X_test_verginica[1], marker='.', label="Iris-Virginica")
        plt.scatter( x=X_test_non_verginica[0], y=X_test_non_verginica[1], marker='>', label="Not Iris-Virginica")
        plt.show()

logisticRegression_Iris = LogisticRegression_Iris()
frame_train_x, train_y, frame_test_x, test_y = logisticRegression_Iris.get_data_sets()


verginica_pipeline = logisticRegression_Iris.create_pipeline_verginica(train_y)
petal_pipeline = logisticRegression_Iris.create_pipeline_petal(frame_train_x)

train_y_verginica = logisticRegression_Iris.runthrough_pipeline(verginica_pipeline, train_y)


frame_train_x_verginica__Petal_width = logisticRegression_Iris.runthrough_pipeline(petal_pipeline, frame_train_x)


import numpy as np
# frame_train_x_verginica__Petal_width = np.array(frame_train_x_verginica__Petal_width.tolist()).reshape(len(frame_train_x_verginica__Petal_width.tolist()), 1)
frame_train_x_verginica__Petal_width = frame_train_x_verginica__Petal_width.to_numpy()
train_y_verginica = train_y_verginica.tolist()


print(frame_train_x_verginica__Petal_width)
print(train_y_verginica)


feature_list = logisticRegression_Iris.get_all_features(petal_pipeline, frame_train_x)


print("\n")
print("All feature labels : \n{}".format(feature_list))
print("\n")
print(f"Train set x length : %d || Train set y length : %d", len(frame_train_x_verginica__Petal_width), len(train_y_verginica))
print("\n")
# print(f"Test  set x length : %d || Test  set y length : %d", len(test_x), len(test_y))
print("\n")

log_Reg = logisticRegression_Iris.Perform_VerginicaDetection(frame_train_x_verginica__Petal_width, train_y_verginica)


test_y_verginica = logisticRegression_Iris.runthrough_pipeline(verginica_pipeline, test_y)


frame_test_x_verginica__Petal_width = logisticRegression_Iris.runthrough_pipeline(petal_pipeline, frame_test_x)


import numpy as np
# frame_test_x_verginica__Petal_width = np.array(frame_test_x_verginica__Petal_width.tolist()).reshape(len(frame_test_x_verginica__Petal_width.tolist()), 1)
frame_test_x_verginica__Petal_width = frame_test_x_verginica__Petal_width.to_numpy()

test_y_verginica = test_y_verginica.tolist()


print(frame_test_x_verginica__Petal_width)
print(test_y_verginica)

logisticRegression_Iris.Predict_VerginicaPetalBasedDetection(log_Reg, frame_test_x_verginica__Petal_width, test_y_verginica)

