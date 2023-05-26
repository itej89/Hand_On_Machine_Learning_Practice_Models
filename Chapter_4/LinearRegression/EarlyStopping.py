import os


class EarlyStopping:



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
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.preprocessing import StandardScaler
        
        
        num_pipeline = Pipeline([
            ('poly_features', PolynomialFeatures(degree=90, include_bias=False)),
            ('std_Scaler', StandardScaler()),
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
        return ['Bias'] + num_attribs

    #using closed-form solution to Eq 4-12
    def Perform_SGDEarlyStopping(self, X, y, X_test, y_test):
        import numpy as np
        from sklearn.linear_model import SGDRegressor
        sgd_reg = SGDRegressor(max_iter=1, tol=None, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
        min_val_error = float('inf')
        best_epoch = None
        best_model = None
        for epoch in range(10000):
            sgd_reg.fit(X,y)
            y_val_pred = sgd_reg.predict(X_test)

            from sklearn.metrics import mean_squared_error
            from sklearn.base import clone
            val_error = mean_squared_error(y_test, y_val_pred)
            if val_error < min_val_error:
                min_val_error = val_error
                best_epoch = epoch
                best_model = clone(sgd_reg)
                print(f"Epoch : {epoch} , Min val error : {min_val_error}")



earlyStopping = EarlyStopping()
frame_train_x, train_y, frame_test_x, test_y = earlyStopping.get_data_sets()
full_pipeline = earlyStopping.create_pipeline(frame_train_x)
train_x = earlyStopping.runthrough_pipeline(full_pipeline, frame_train_x)
test_x = earlyStopping.runthrough_pipeline(full_pipeline, frame_test_x)

feature_list = earlyStopping.get_all_features(full_pipeline, frame_train_x)


print("\n")
print("All feature labels : \n{}".format(feature_list))
print("\n")
print(f"Train set x length : %d || Train set y length : %d", len(train_x), len(train_y))
print("\n")
print(f"Test  set x length : %d || Test  set y length : %d", len(test_x), len(test_y))
print("\n")

lin_reg = earlyStopping.Perform_SGDEarlyStopping(train_x, train_y, test_x, test_y)