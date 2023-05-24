import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import  RandomForestRegressor

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from sklearn.metrics import explained_variance_score

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
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/housing/housing.csv"))
        feat_cols = list(df.columns[:-3]) + [df.columns[-1]]
        target_col = df.columns[-2]

        return df, feat_cols, target_col

    def split_data(self, df, feat_cols, target_col):
        """Splits data into test and train

        Args:
            df (pandas frame): pandas frame
            feat_cols (column name list): feature columns
            target_col (target column): target column name

        Returns:
            tuple: test data frame and target series
        """
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.2)
        return train, test

    def get_column_transormer(self):
        """Function to create transformations for non numerical data and 
        numerical data preprocessing and data cleaning

        Returns:
            sklearn transformer: data transformer
        """
        ct = ColumnTransformer([("impute", SimpleImputer(missing_values=np.nan, strategy='mean'), make_column_selector(dtype_include=np.number)), 
                                # ("std_scaler", StandardScaler(), make_column_selector(dtype_include=np.number)), 
                                ("ordinal_encoder", OrdinalEncoder(), make_column_selector(pattern="ocean_proximity", dtype_include=object))
                                ])
        return ct

    def create_param_search_pipeline(self):
        """Function creates pipeline to search parameter space

        Returns:
            pipeline: sklearn pipeline
        """
        param_grid = {'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8, 10], 'max_depth':[10,30,50]}
        forest_reg = RandomForestRegressor()
        grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=5,
                    scoring='neg_mean_squared_error',return_train_score=True)


        pipeline = make_pipeline(self.get_column_transormer(), grid_search)
        return pipeline

    def create_model_pipeline(self):
        """Funciton creates pipeline that can model the data

        Returns:
            pipeline: sklearn pipeline
        """
        pipeline = make_pipeline(self.get_column_transormer(), RandomForestRegressor(n_estimators=100,max_depth=30, max_features=6))

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
        lin_mse = mean_squared_error(test[target_col], pred)
        print(f"MSE : {lin_mse}")

        R_square = r2_score(test[target_col], pred)
        print('Coefficient of Determination', R_square)

        result=explained_variance_score(test[target_col], pred,multioutput='uniform_average')
        print('explained_variance_score', result) 

    
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


