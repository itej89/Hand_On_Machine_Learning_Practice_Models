import os, sys

from pathlib import Path 




from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS

class kmeans_iris_model :

    def split_data(self, pandas_frame, stratification_column_id, target_column_id):
        from Splitdata import SplitData
        objSplitData = SplitData()

        strat_train_set, strat_test_set = objSplitData.stratified_split_data(_panda_data_frame=pandas_frame, _n_splits=1, _test_percent=0.2, _randome_state=42, _columnID=stratification_column_id)   

        strat_train_set_data = strat_train_set.drop(target_column_id, axis=1)
        strat_train_set_labels = strat_train_set[[target_column_id]]

        strat_test_set_data = strat_test_set.drop(target_column_id, axis=1)
        strat_test_set_labels = strat_test_set[target_column_id]

        return strat_train_set_data, strat_train_set_labels, strat_test_set_data, strat_test_set_labels

    def get_data_sets(self):
        from pathlib import Path
        
        DATA_PATH = os.path.join(pwd.parent, DATASET_PATH_CONSTANTS.DIR, DATASET_PATH_CONSTANTS.NAME)

        #Load Data
        from LoadData import LoadData
        objLoadData = LoadData()
        data_frame = objLoadData.load_Data(os.path.join(DATA_PATH, DATASET_PATH_CONSTANTS.BLOB), False)
      
        train_x , train_y, test_x, test_y = self.split_data(data_frame, "target", "target")
        print(train_y)
        return train_x.to_numpy(dtype = 'u1') , train_y.to_numpy(dtype = 'i').ravel(), test_x.to_numpy(dtype = 'u1'), test_y.to_numpy(dtype = 'i').ravel()
        

    
    def create_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
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


    def get_representative_digits(self, cluste_count, train_x, train_y):
        from sklearn.cluster import KMeans
        import numpy as np
        k = cluste_count
        kmeans = KMeans(n_clusters=k)
        X_digits_dist = kmeans.fit_transform(train_x)
        representative_digit_idx = np.argmin(X_digits_dist, axis=0)
        X_representative_digits = train_x[representative_digit_idx]

        #This should be performed manually
        Y_representative_digits = train_y[representative_digit_idx]

        return X_representative_digits, Y_representative_digits


    def build_model(self):
        from sklearn.linear_model import LogisticRegression
        logisticRegression = LogisticRegression()
        return logisticRegression



    def fit_model(self, model, train_X, train_y):
        # Start/resume training
        model.fit(train_X, train_y)
        return model

    def display_inertia(self, train_x):
        from sklearn.cluster import KMeans
        Error =[]
        for i in range(1, 11):
            model = KMeans(n_clusters = i).fit(train_x)
            model.fit(train_x)
            Error.append(model.inertia_)
        import matplotlib.pyplot as plt
        plt.plot(range(1, 11), Error)
        plt.title('Elbow method')
        plt.xlabel('No of clusters')
        plt.ylabel('Error')
        plt.show()


    def print_model_params(self, model, cluster_count ,train_X):
        print("---------------------------------------------------------------")
        print(f"Model Params for cluster_count : {cluster_count}")
        print("---------------------------------------------------------------")
        print(f"Labels : {model.labels_}")
        print(f"Centroids : {model.cluster_centers_}")
        print(f"Score : {model.score(train_X)}")
        from sklearn.metrics import silhouette_score

        #Hight Silhouette Score means better model
        print(f"Silhouette Score : {silhouette_score(train_X, model.labels_)}")
        print("---------------------------------------------------------------")
        print("                                                                ")


    def print_model_score(self, model, test_X, test_y):
        print(f"Score : {model.score(test_X, test_y)}")   

    def model_predict(self, model, test_x):
        print(f"Predicted classes : {model.predict(test_x)}")
   


kmeans_iris_model = kmeans_iris_model() 
train_x , train_y, test_x, test_y = kmeans_iris_model.get_data_sets()


# input_pipeline = kmeans_iris_model.create_pipeline()

# train_x = kmeans_iris_model.runthrough_pipeline(input_pipeline, train_x)
# test_x = kmeans_iris_model.runthrough_pipeline(input_pipeline, test_x)


significant_train_x, manually_labeled_train_y = kmeans_iris_model.get_representative_digits(50, train_x, train_y)


#Look fo relbow to get the idea number of clusters for the dataset
# kmeans_iris_model.display_inertia(train_x)

iris_model = kmeans_iris_model.build_model()

iris_model = kmeans_iris_model.fit_model(iris_model, significant_train_x, manually_labeled_train_y)

# iris_model = kmeans_iris_model.fit_model(iris_model, train_x, train_y)

kmeans_iris_model.print_model_score(iris_model, test_x, test_y)




# In the KMeans class, the transform() method measures the
# distance from each instance to every centroid:
# print(iris_model.transform(test_x))

# kmeans_iris_model.model_predict(iris_model, test_x)