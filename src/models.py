from sklearn.ensemble import IsolationForest
import numpy as np


class IsolationForestModel:
    def __init__(self, contamination=0.01, n_estimators=100, max_samples='auto'):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X):
        self.model.fit(X)

    def get_scores(self, X):
        
       raw_scores = -self.model.decision_function(X)
       scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
       return scores

    def predict(self, X, threshold):
        scores = self.get_scores(X)
        return (scores > threshold).astype(int)
    
from sklearn.neighbors import LocalOutlierFactor


from sklearn.neighbors import LocalOutlierFactor
import pandas as pd


from sklearn.neighbors import LocalOutlierFactor
import numpy as np


class LOFModel:
    def __init__(self, n_neighbors=10, contamination=0.02):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )

    def train(self, X):
        self.model.fit(X)

    def get_scores(self, X):
        scores = -self.model.score_samples(X)

        # Normalize safely
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

        return scores

    def predict(self, X, threshold):
        scores = self.get_scores(X)
        return (scores > threshold).astype(int)