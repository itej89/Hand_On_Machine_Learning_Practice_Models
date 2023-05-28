from sklearn.base import BaseEstimator, TransformerMixin


class Verginica_Label_Filter_Transformer(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        import numpy as np
        return (X == 2).astype(np.int)