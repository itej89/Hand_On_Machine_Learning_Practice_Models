from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class BiasInsert_Transformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room =True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        import numpy as np
        return np.c_[np.ones((len(X),1)), X]