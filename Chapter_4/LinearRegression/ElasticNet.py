import os


class ElasticNet:

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

    #using closed-form solution to Eq 4-12
    def Perform_ElasticNet(self, X, y, X_test):
        from sklearn.linear_model import ElasticNet
        elasticNet = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elasticNet.fit(X,y)
        print(elasticNet.predict(X_test))



elasticNet = ElasticNet()
frame_train_x, train_y, frame_test_x, test_y = elasticNet.get_data_sets()
full_pipeline = elasticNet.create_pipeline(frame_train_x)
train_x = elasticNet.runthrough_pipeline(full_pipeline, frame_train_x)
test_x = elasticNet.runthrough_pipeline(full_pipeline, frame_test_x)

feature_list = elasticNet.get_all_features(full_pipeline, frame_train_x)


print("\n")
print("All feature labels : \n{}".format(feature_list))
print("\n")
print(f"Train set x length : %d || Train set y length : %d", len(train_x), len(train_y))
print("\n")
print(f"Test  set x length : %d || Test  set y length : %d", len(test_x), len(test_y))
print("\n")

lin_reg = elasticNet.Perform_ElasticNet(train_x, train_y, test_x)