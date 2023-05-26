import os, sys

from pathlib import Path 

pwd = Path(os.path.abspath(__file__))


from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS

class svm_iris_model :

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


    def build_linear_svc_soft_margin_Model(self):
        from sklearn.svm import LinearSVC
        lin_svc = LinearSVC(C=1, loss="hinge")
        return lin_svc


    def fit_model(self, model, train_X, train_y):
        # Start/resume training
        model.fit(train_X, train_y)
        return model


    def model_predict(self, model, test_x):
        print(f"Predicted classes : {model.predict(test_x)}")
   


svm_iris_model = svm_iris_model() 
train_x , train_y, test_x, test_y = svm_iris_model.get_data_sets()


input_pipeline = svm_iris_model.create_pipeline()

train_x = svm_iris_model.runthrough_pipeline(input_pipeline, train_x)
test_x = svm_iris_model.runthrough_pipeline(input_pipeline, test_x)

iris_model = svm_iris_model.build_linear_svc_soft_margin_Model()

iris_model = svm_iris_model.fit_model(iris_model, train_x , train_y)

svm_iris_model.model_predict(iris_model, test_x)