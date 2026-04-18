import pickle
import pandas as pd
import numpy as np
import os

class FraudDetector:
    def __init__(self, model_path="models/iso_model.pkl"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        full_path = os.path.join(project_root, model_path) 
        with open(full_path, "rb") as f:
            self.model = pickle.load(f)
    def score(self, features_list):
      cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
      df = pd.DataFrame([features_list], columns=cols)
      df['Amount_log'] = np.log1p(df['Amount'])
      df['Hour'] = 14 
      raw_score = -self.model.decision_function(df)[0]
      score = (raw_score + 0.5) / 1.0
      return max(0, min(1, score))

    def get_action(self, score):
        if score < 0.3:
            return "ALLOW"
        elif score < 0.5:
            return "ALERT"
        elif score < 0.7:
            return "REVIEW"
        else:
            return "BLOCK"