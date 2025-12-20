import numpy as np
from alfa_dbasmote import AlphaDistanceDBASMOTE
from ar_adasyn import ARADASYN
from pc_smote import PCSMOTE

class AlphaDistanceDBASMOTEWrapper:
    def __init__(self, beta=1.0, m=5, random_state=None):
        self.beta = beta
        self.m = m
        self.random_state = random_state

    def fit_resample(self, X_min, X_maj):
        X_syn = AlphaDistanceDBASMOTE(X_min, X_maj, beta=self.beta, m=self.m, random_state=self.random_state)
        X_resampled = np.vstack([X_min, X_maj, X_syn])
        y_resampled = np.array([1]*len(X_min) + [0]*len(X_maj) + [1]*len(X_syn))
        return X_resampled, y_resampled


class ARADASYNWrapper:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = random_state

    def fit_resample(self, X_min, X_maj):
        X_syn = ARADASYN(X_min, X_maj, k=self.k, random_state=self.random_state)
        X_resampled = np.vstack([X_min, X_maj, X_syn])
        y_resampled = np.array([1]*len(X_min) + [0]*len(X_maj) + [1]*len(X_syn))
        return X_resampled, y_resampled

class PCSMOTEWrapper:
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampler = PCSMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)

    def fit_resample(self, X, y):
        return self.sampler.fit_resample(X, y)
