import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_thresholds(df, score_col='iso_score'):

    y_true = df['Class']
    scores = df[score_col]

    thresholds = np.linspace(0, 1, 50)

    results = []

    for t in thresholds:
        y_pred = (scores > t).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return pd.DataFrame(results)


def print_best_thresholds(results_df):
    print("\n🔍 BEST BY F1:")
    print(results_df.sort_values('f1', ascending=False).head(3))

    print("\n🔍 HIGH RECALL (>=80%):")
    print(results_df[results_df['recall'] >= 0.8].head(3))

    print("\n🔍 HIGH PRECISION (>=80%):")
    print(results_df[results_df['precision'] >= 0.8].head(3))