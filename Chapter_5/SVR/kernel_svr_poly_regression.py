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


    def build_model(self):
        from sklearn.svm import LinearSVR
        model = LinearSVR(epsilon=1.5)
        return model

    def fit_model(self, model, train_X, train_y):
        # Start/resume training
        model.fit(train_X, train_y)
        return model

    def model_predict(self, model, test_x):
        print(f"Predicted value : {model.predict(test_x)}")
   


cal_housing_model = svr_cal_housing_model() 
train_x , train_y, test_x, test_y, validation_x, validation_y = cal_housing_model.get_data_sets()

input_pipeline = cal_housing_model.create_pipeline()
train_x = cal_housing_model.runthrough_pipeline(input_pipeline, train_x)
test_x = cal_housing_model.runthrough_pipeline(input_pipeline, test_x)

model = cal_housing_model.build_model()
model = cal_housing_model.fit_model(model, train_x , train_y)
cal_housing_model.model_predict(model, test_x)
print(test_y.to_numpy())