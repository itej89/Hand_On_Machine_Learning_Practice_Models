from sklearn.preprocessing import Normalizer
import numpy as np

class Normalize3D(Normalizer):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)