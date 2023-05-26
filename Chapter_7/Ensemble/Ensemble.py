import os, sys

from pathlib import Path 




from CommonConstants import DATASET_PATH_CONSTANTS, MODEL_PATH_CONSTANTS, LABELS

class tree_iris_model :

    def __init__(self) -> None:
        self.feature_names = None
        self.target_names = None

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
        self.feature_names = objLoadData.get_feature_names()
        self.target_names = objLoadData.get_target_names()
        train_x , train_y, text_x, test_y = self.split_data(data_frame)
        
        return train_x , train_y, text_x, test_y

    
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




    def build_voting_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm  import SVC

        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()

        voting_clf = VotingClassifier(
            estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
            voting="hard")

        return voting_clf

    def build_bagging_model(self):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier

        bag_clf = BaggingClassifier(DecisionTreeClassifier(),
        n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)

        return bag_clf

    def build_random_forests_model(self):
        from sklearn.ensemble import RandomForestClassifier

        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

        return rnd_clf

    def build_ada_boost_model(self):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier

        rnd_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                    algorithm="SAMME.R", learning_rate=0.5)

        return rnd_clf
        



    def fit_model(self, model, train_X, train_y):
        # Start/resume training
        model.fit(train_X, train_y)
        return model

    def export_model_graph(self, model ):
        from sklearn.tree import export_graphviz
        export_graphviz(
        model,
        out_file=MODEL_PATH_CONSTANTS.GetModelGraphPath(),
        feature_names=self.feature_names,
        class_names=self.target_names,
        rounded=True,
        filled=True
        )

    def print_bagging_oob_score(self, model):
        print("oob_Score : {}".format(model.oob_score_))

    def print_rnd_forest_score_score(self, model):
        for name, score in zip(['Sepel len', 'Sepel Width', 'Petal len', 'Petal Width'], model.feature_importances_):
            print(name, score)

    def model_predict(self, model, test_x):
        print(f"Predicted classes : {model.predict(test_x)}")

    def model_predict_proba(self, model, features):
        print(f"Predicted classes : {model.predict_proba(features)}")
   


tree_iris_model = tree_iris_model() 
train_x , train_y, test_x, test_y = tree_iris_model.get_data_sets()


input_pipeline = tree_iris_model.create_pipeline()

train_x = tree_iris_model.runthrough_pipeline(input_pipeline, train_x)
test_x = tree_iris_model.runthrough_pipeline(input_pipeline, test_x)

iris_model = tree_iris_model.build_ada_boost_model()

iris_model = tree_iris_model.fit_model(iris_model, train_x , train_y)

# tree_iris_model.export_model_graph(iris_model)

tree_iris_model.model_predict(iris_model, test_x)


# tree_iris_model.print_rnd_forest_score_score(iris_model)

# tree_iris_model.print_bagging_oob_score(iris_model)

# tree_iris_model.model_predict_proba(iris_model, [[0.66, 0.2, 0.66, 0.4]])