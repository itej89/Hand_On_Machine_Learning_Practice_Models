import os, sys

from pathlib import Path 




from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS

class kmeans_iris_model :

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
        pwd = Path(os.path.abspath(__file__))
        DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data_frame = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)

        train_x , train_y, text_x, test_y = self.split_data(data_frame)
        
        return train_x , train_y, text_x, test_y

    
    def create_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LogisticRegression
        
        input_pipeline = Pipeline([
            ('std_Scaler', StandardScaler()),
            ("kmeans", KMeans(n_clusters=3))
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


    def grid_Search(self, pipeline, train_X, train_y):
        from sklearn.model_selection import GridSearchCV
        param_grid = dict(kmeans__n_clusters=range(2, 20))
        grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
        grid_clf.fit(train_X, train_y)
        print(f"Grid search best params : {grid_clf.best_params_}")


    def build_model(self, cluster_count = 3):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cluster_count)
        return kmeans
 

    def fit_model(self, model, train_X, train_y):
        # Start/resume training
        model.fit(train_X, train_y)
        return model

    

    def print_model_score(self, model, test_X, test_y):
        print(f"Score : {model.score(test_X, test_y)}")    

    def model_predict(self, model, test_x):
        print(f"Predicted classes : {model.predict(test_x)}")
   


kmeans_iris_model = kmeans_iris_model() 
train_x , train_y, test_x, test_y = kmeans_iris_model.get_data_sets()


input_pipeline = kmeans_iris_model.create_pipeline()

# iris_model = kmeans_iris_model.grid_Search(input_pipeline, train_x, train_y)
iris_model = kmeans_iris_model.fit_model(input_pipeline, train_x, train_y)

kmeans_iris_model.print_model_score(iris_model, test_x, test_y)



# In the KMeans class, the transform() method measures the
# distance from each instance to every centroid:
# print(iris_model.transform(test_x))

# kmeans_iris_model.model_predict(iris_model, test_x)