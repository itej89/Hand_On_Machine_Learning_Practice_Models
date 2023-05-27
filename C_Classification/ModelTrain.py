import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.metrics import precision_score, recall_score, precision_recall_curve ,f1_score, roc_curve


import matplotlib.pyplot as plt

#Dummy class to print data in the middle of sklearn pipeline
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(X)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
#-------------------------------------------------------------


#Estimator class
class Trainer:

    def LoadData(self):
        """Load Data

        Returns:
            tuple: returns dataframe, feature columns, target columns
        """
        file_object = open(os.path.join(os.path.abspath(''), "datasets/mnist/mnist_dataset.pkl"), "rb")
        mnist = pickle.load(file_object)

        df = pd.DataFrame(data= np.c_[mnist['data'], mnist['target']],
                            columns= mnist['feature_names'] + ['target'])
        
        feat_cols = list(df.columns[:-1])
        target_cols = df.columns[-1]
        
        return df, feat_cols, target_cols

    def split_data(self, df, feat_cols, target_col):
        """Splits data into test and train

        Args:
            df (pandas frame): pandas frame
            feat_cols (column name list): feature columns
            target_col (target column): target column name

        Returns:
            tuple: test data frame and target series
        """
        strsplt = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train = None
        test = None

        for i, (train_idx, test_idx) in enumerate(strsplt.split(df[feat_cols], df[target_col])):
            print(train_idx)
            print(test_idx)
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

        return train, test

    def create_param_search_pipeline(self):
        """Function creates pipeline to search parameter space

        Returns:
            pipeline: sklearn pipeline
        """
        param_grid = {'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8, 10], 'max_depth':[10,30,50]}
        sgdClassifier = SGDClassifier()
        grid_search = RandomizedSearchCV(sgdClassifier, param_grid, cv=5,
                    scoring='neg_mean_squared_error',return_train_score=True)

        pipeline = make_pipeline(grid_search)
        return pipeline

    def create_model_pipeline(self):
        """Funciton creates pipeline that can model the data

        Returns:
            pipeline: sklearn pipeline
        """
        pipeline = make_pipeline(SGDClassifier())

        return pipeline
    
    def fit(self, pipeline, X, y):
        """Fits data to the pipeline

        Args:
            pipeline (sklearn pipeline): pipeline object
            X (pandas frame): Train X
            y (pandas frame): Train y

        Returns:
            pipeline: trained model
        """
        pipeline.fit(X, y)
        return pipeline
    
 
    
    def print_scores(self, pred, actual):
        """Prints multiple metrics for regression

        Args:
            pred (numpy array): model predictions
            actual (numpy array): test data
        """

        prec_score = precision_score(actual, pred, average=None)
        rec_score = recall_score(actual, pred, average=None)
        rec_score = recall_score(actual, pred, average=None)
        f1_scores = f1_score(actual, pred, average=None)

        print("Precision Score : {}".format(prec_score))
        print("Recall Score : {}".format(rec_score))
        print("f1 score : {}".format(f1_scores))



if __name__ =="__main__":
    _trainer = Trainer()
    df, feat_cols, target_col = _trainer.LoadData()
    train, test = _trainer.split_data(df, feat_cols, target_col)

    #Find best model parameters----------------------------------------------
    # pipeline = _trainer.create_param_search_pipeline()
    # pipeline = _trainer.fit(pipeline, train[feat_cols], train[target_col])
    # print(pipeline[-1].best_params_)
    #------------------------------------------------------------------------

    #Fit best model and run prediction---------------------------------------
    pipeline = _trainer.create_model_pipeline()
    pipeline = _trainer.fit(pipeline, train[feat_cols], train[target_col])
    pred = pipeline.predict(test[feat_cols])
    _trainer.print_scores(pred, test[target_col])
    #------------------------------------------------------------------------
