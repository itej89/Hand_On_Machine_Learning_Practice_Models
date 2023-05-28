from sklearn.base import BaseEstimator, TransformerMixin


class Petal_Width_Filter_Transformer(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X["petal width (cm)"]