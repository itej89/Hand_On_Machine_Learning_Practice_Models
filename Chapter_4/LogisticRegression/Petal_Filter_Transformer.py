from sklearn.base import BaseEstimator, TransformerMixin


class Petal_Filter_Transformer(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.drop("petal width (cm)", axis=1)
        X = X.drop("petal length (cm)", axis=1)
        return X