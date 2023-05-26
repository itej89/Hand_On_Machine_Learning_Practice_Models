import os, sys

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd.parent, "KerasCallbacks"))
sys.path.append(os.path.join(pwd.parent, "PipelineTransformers"))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS

class svr_cal_housing_model :


    def split_data(self, pandas_frame):
     
        from Splitdata import SplitData
        objSplitData = SplitData()

        strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")
        
        strat_train_set.reset_index(inplace = True)
        strat_train_set, strat_val_set = objSplitData.stratified_split_data(_panda_data_frame=strat_train_set, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID="income_category")


        strat_train_set = strat_train_set.drop("income_category", axis=1)
        strat_test_set = strat_test_set.drop("income_category", axis=1)
        strat_val_set = strat_val_set.drop("income_category", axis=1)

        strat_train_set_data = strat_train_set.drop("index", axis=1)
        strat_train_set_data = strat_train_set_data.drop("target", axis=1)
        strat_train_set_labels = strat_train_set["target"]

        strat_test_set_data = strat_test_set.drop("target", axis=1)
        strat_test_set_labels = strat_test_set["target"]

        strat_val_set_data = strat_val_set.drop("index", axis=1)
        strat_val_set_data = strat_val_set_data.drop("target", axis=1)
        strat_val_set_labels = strat_val_set["target"]

        return strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels,strat_test_set_data, strat_test_set_labels


    def get_data_sets(self):
        from pathlib import Path
        pwd = Path(os.path.abspath(__file__))
        DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)

        import numpy as np
        data = objLoadData.categorize_value_column(data, "MedInc", "income_category", \
        _bins=[0.,1.5,3.0,4.5,6.0,np.inf], _labels=[1,2,3,4,5])

        strat_train_set_data, strat_train_set_labels, strat_val_set_data, strat_val_set_labels, strat_test_set_data, strat_test_set_labels = self.split_data(data)

        return strat_train_set_data, strat_train_set_labels, strat_test_set_data, strat_test_set_labels,strat_val_set_data, strat_val_set_labels

    
    def create_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        input_pipeline = Pipeline([
            ('std_Scaler', StandardScaler())
        ])

        return input_pipeline
    
    def runthrough_pipeline(self, full_pipeline, data_frame):
        data_formated = full_pipeline.fit_transform(data_frame)
        return data_formated

    


    def get_num_attributes(self, data_frame):
        num_attribs = list(data_frame)
        return num_attribs

    def get_all_features(self, full_pipeline, data_frame):
        num_attribs = self.get_num_attributes(data_frame)
        return num_attribs


    def build_xgboost_model(self):
        from xgboost  import XGBRegressor
        return XGBRegressor()

    def build_GradientBoostingRegressor_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)

    def gradboostreg_fit_model(self, gbrt, train_X, train_y, validation_x, validation_y):
        from sklearn.metrics import mean_squared_error
        min_val_error = float("inf")
        error_going_up = 0
        for n_estimators in range(1, 120):
            gbrt.n_estimators = n_estimators
            gbrt.fit(train_X, train_y)
            y_pred = gbrt.predict(validation_x)
            val_error = mean_squared_error(validation_y, y_pred)
            if val_error < min_val_error:
                min_val_error = val_error
                error_going_up = 0
            else:
                error_going_up += 1
            if error_going_up == 5:
                break # early stopping
        return gbrt


    def PCA_fit_model(self, model, train_X, train_y):
        from sklearn.decomposition import PCA
        # pca = PCA(n_components=0.98) #PCA with total varanice of all dimensions (Use this to get the nubmer of components)
        pca = PCA(n_components=6, svd_solver="randomized") # PCA with number of output dimensions, svd_solver="full"
        X4D = pca.fit_transform(train_X) 
        RECOVERED_X = pca.inverse_transform(X4D)
        print(f"X : {train_X[0]}")
        print(f"X4D : {X4D[0]}")
        print(f"X` : {RECOVERED_X[0]}")

        model.fit(X4D, train_y)
        return pca, model

    def PCA_model_predict(self, model, pca, test_x):
        X4D = pca.fit_transform(test_x)
        print(f"Predicted value : {model.predict(X4D)}")


    def INCREMENTAL_fit_model(self, model, train_X, train_y):
        from sklearn.decomposition import IncrementalPCA
        # pca = PCA(n_components=0.98) #PCA with total varanice of all dimensions (Use this to get the nubmer of components)
        incrementalPCA = IncrementalPCA(n_components=6) # PCA with number of output dimensions, svd_solver="full"
        n_batches = 100
        import numpy as np
        for X_batch in np.array_split(train_X, n_batches):
            incrementalPCA.partial_fit(X_batch) 

        X4D = incrementalPCA.transform(train_X)
        RECOVERED_X = incrementalPCA.inverse_transform(X4D)
        print(f"train_X : {train_X[0]}")
        # print(f"train_X4D : {X4D[0]}")
        print(f"train_X` : {RECOVERED_X[0]}")

        model.fit(X4D, train_y)
        return incrementalPCA, model


    def IncrementalPCA_model_predict(self, model, incrementalPCA, test_x):
        incrementalPCA.partial_fit(test_x)
        X4D = incrementalPCA.transform(test_x)
        RECOVERED_X = incrementalPCA.inverse_transform(X4D)
        print(f"test_X : {test_x[0]}")
        # print(f"test_X4D : {X4D[0]}")
        print(f"test_X` : {RECOVERED_X[0]}")
        print(f"Predicted value : {model.predict(X4D)}")


    def kernle_grid_search_model(self, model, train_X, train_y):
        from sklearn.decomposition import KernelPCA
        from sklearn.pipeline import Pipeline
        import numpy as np
        clf = Pipeline([
                ("kpca", KernelPCA(n_components=2)),
                ("model", model)
            ])
        param_grid = [{
                "kpca__gamma": np.linspace(0.03, 0.05, 10),
                "kpca__kernel": ["rbf", "sigmoid"]
            }]
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(clf, param_grid, cv=3)
        grid_search.fit(train_X, train_y)
        print(grid_search.best_params_)

    #above kernle_grid_search_model function generated these values
    # {'kpca__gamma': 0.03666666666666667, 'kpca__kernel': 'sigmoid'}
    def kernel_fit_model(self, model, train_X, train_y):
        from sklearn.decomposition import KernelPCA
        # kernelPCA = KernelPCA(n_components=0.98) #PCA with total varanice of all dimensions (Use this to get the nubmer of components)
        kernelPCA = KernelPCA(n_components=6, kernel="sigmoid", gamma=0.037, fit_inverse_transform = True) # PCA with number of output dimensions
        X4D = kernelPCA.fit_transform(train_X) 
        RECOVERED_X = kernelPCA.inverse_transform(X4D)
        print(f"X : {train_X[0]}")
        print(f"X4D : {X4D[0]}")
        print(f"X` : {RECOVERED_X[0]}")

        model.fit(X4D, train_y)
        return kernelPCA, model

    def kernel_model_predict(self, model, kernel_pca, test_x):
        X4D = kernel_pca.fit_transform(test_x)
        print(f"Predicted value : {model.predict(X4D)}")


    def LLE_fit_model(self, model, train_X, train_y):
        from sklearn.manifold import LocallyLinearEmbedding
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
        X4D = lle.fit_transform(train_X) 

        model.fit(X4D, train_y)
        return lle, model

    def LLE_model_predict(self, model, lle, test_x):
        X4D = lle.fit_transform(test_x)
        print(f"Predicted value : {model.predict(X4D)}")


cal_housing_model = svr_cal_housing_model()
train_x , train_y, test_x, test_y, validation_x, validation_y = cal_housing_model.get_data_sets()

input_pipeline = cal_housing_model.create_pipeline()
train_x = cal_housing_model.runthrough_pipeline(input_pipeline, train_x)
test_x = cal_housing_model.runthrough_pipeline(input_pipeline, test_x)

model = cal_housing_model.build_xgboost_model()
# cal_housing_model.kernle_grid_search_model(model, train_x , train_y)
pca, model = cal_housing_model.kernel_fit_model(model, train_x , train_y)
cal_housing_model.kernel_model_predict(model, pca, test_x)
print(f"Actual value : {test_y.to_numpy()}")