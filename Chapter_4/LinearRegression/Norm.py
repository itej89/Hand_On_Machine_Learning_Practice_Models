import os


class NormRegression:



    def split_data(self, pandas_frame):
        from Splitdata import SplitData
        objSplitData = SplitData()

        strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="y_cat")   

        strat_train_set = strat_train_set.drop("y_cat", axis=1)
        strat_test_set = strat_test_set.drop("y_cat", axis=1)

        strat_train_set_data = strat_train_set.drop("y", axis=1)
        strat_train_set_labels = strat_train_set["y"]

        strat_test_set_data = strat_test_set.drop("y", axis=1)
        strat_test_set_labels = strat_test_set["y"]

        return strat_train_set_data, strat_train_set_labels, strat_test_set_data, strat_test_set_labels

    def get_data_sets(self):
        from pathlib import Path
        pwd = Path(os.path.abspath(__file__))
        DATA_PATH = os.path.join(pwd.parent, "datasets", "linear_rand")

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data_frame = objLoadData.load_Data(os.path.join(DATA_PATH, "linear_rand.pkl"), False)

        import numpy as np
        data_frame = objLoadData.categorize_value_column(data_frame, "y", "y_cat", \
        _bins=[0.,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,np.inf], _labels=[1,2,3,4,5,6,7,8,9,10])

        train_x , train_y, text_x, test_y = self.split_data(data_frame)
        
        return train_x , train_y, text_x, test_y

    def create_pipeline(self, data_frame):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from BiasInsert_Transformer import BiasInsert_Transformer
        
        num_pipeline = Pipeline([
            # ('std_Scaler', StandardScaler()),
            ('bias_Inserter', BiasInsert_Transformer()),
        ])

        # num_attribs = list(data_frame)


        # from sklearn.compose import ColumnTransformer

        # full_pipeline = ColumnTransformer([
        #     ("num", num_pipeline, num_attribs)
        # ])

        return num_pipeline
    
    def runthrough_pipeline(self, full_pipeline, data_frame):
        data_formated = full_pipeline.fit_transform(data_frame)
        return data_formated


    def get_num_attributes(self, data_frame):
        num_attribs = list(data_frame)
        return num_attribs

    def get_all_features(self, full_pipeline, data_frame):
        num_attribs = self.get_num_attributes(data_frame)
        return ['Bias'] + num_attribs

    def Perform_LinearRegression(self, X, y):
        from sklearn.linear_model import LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X,y)
        print(f"Intercept : {lin_reg.intercept_}, Coefficient : {lin_reg.coef_}")
        return lin_reg

    def Predict_LinearRegression_WithBias(self, lin_reg, X_test, y_test):
        y_pred  = lin_reg.predict(X_test)
        print("Prediction : ")

        import numpy as np
        import pandas as pd
        X_test = np.array(X_test)
        X_test = X_test.T
        df = pd.DataFrame(data=np.c_[X_test[0] ,X_test[1], y_test, y_pred], columns=['Bias', 'X', 'y_act', 'y_pred' ])
        print(df)
        import matplotlib.pyplot as plt
        plt.plot(X_test[1], y_pred, "r-")
        plt.plot(X_test[1], y_test, "b.")
        plt.axis([0, 2, 0, 15])
        plt.show()

    def Predict_LinearRegression_WithoutBias(self, lin_reg, X_test, y_test):
        y_pred  = lin_reg.predict(X_test)
        print("Prediction : ")
        
        import numpy as np
        import pandas as pd
        X_test = np.array(X_test)
        X_test = X_test.reshape(-1)
        df = pd.DataFrame(data=np.c_[X_test, y_test, y_pred], columns=['X', 'y_act', 'y_pred' ])
        print(df)

    

    def numpy_linalg_leastsquares(self, X, y):
        import numpy as np
        theta_best_svd, residuals, rank, s = np.linalg.lstsq(X, y, rcond=1e-6)
        print(f"Linear Regression calculated coeffients : {theta_best_svd}")
    


normRegression = NormRegression()
frame_train_x, train_y, frame_test_x, test_y = normRegression.get_data_sets()
full_pipeline = normRegression.create_pipeline(frame_train_x)
train_x = normRegression.runthrough_pipeline(full_pipeline, frame_train_x)
test_x = normRegression.runthrough_pipeline(full_pipeline, frame_test_x)

feature_list = normRegression.get_all_features(full_pipeline, frame_train_x)


print("\n")
print("All feature labels : \n{}".format(feature_list))
print("\n")
print(f"Train set x length : %d || Train set y length : %d", len(train_x), len(train_y))
print("\n")
print(f"Test  set x length : %d || Test  set y length : %d", len(test_x), len(test_y))
print("\n")

lin_reg = normRegression.Perform_LinearRegression(train_x, train_y)

normRegression.Predict_LinearRegression_WithBias(lin_reg, train_x, train_y.tolist())


# ![](./images/Eq4-3.png)




