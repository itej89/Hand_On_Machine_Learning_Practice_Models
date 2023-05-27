from sklearn.preprocessing import StandardScaler
import numpy as np

class StandardScaler3D(StandardScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)