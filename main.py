import os
import pickle

from src.utils import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
from src.models import IsolationForestModel, LOFModel
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
import joblib

def train_models():
    # Load processed data
    train_df = pd.read_csv("data/train_processed.csv")
    test_df = pd.read_csv("data/test_processed.csv")

    X_train = train_df.drop('Class', axis=1)
    X_test = test_df.drop('Class', axis=1)

    # -------- Isolation Forest --------
    print("\n🚀 Training Isolation Forest...")

    iso_model = IsolationForestModel(
        contamination=0.02,
        n_estimators=300,
        max_samples=512
    )

    iso_model.train(X_train)

    iso_scores = iso_model.get_scores(X_test)

    print("✅ Isolation Forest Done!")

    # -------- LOF --------
    print("\n🚀 Training LOF...")

# PCA for LOF
    pca = PCA(n_components=10)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

# Try ONE value first
    lof_model = LOFModel(
    n_neighbors=10,
    contamination=0.02
)

    lof_model.train(X_train_pca)

    lof_scores = lof_model.get_scores(X_test_pca)

    print("✅ LOF Done!")

    # Save scores
    test_df['iso_score'] = iso_scores
    test_df['lof_score'] = lof_scores

    test_df.to_csv("data/test_scored.csv", index=False)

    print("\n💾 Scores saved to test_scored.csv")
    print("\n🔍 Score Validation:")
    print("ISO Score Range:", iso_scores.min(), "→", iso_scores.max())
    print("LOF Score Range:", lof_scores.min(), "→", lof_scores.max())
    
    os.makedirs("models", exist_ok=True)
    with open("models/iso_model.pkl", "wb") as f:
      pickle.dump(iso_model.model, f)
    

def main():
    print("🚀 SecurePay Fraud Detection Pipeline Started...\n")

    # Step 1: Load Data
    df = load_data("data/creditcard.csv")

    print("✅ Data Loaded Successfully!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())

    # Step 2: Check class distribution
    print("\n📊 Class Distribution:")
    print(df['Class'].value_counts())
    print("\nClass Ratio:")
    print(df['Class'].value_counts(normalize=True))

    # Step 3: Split Data (IMPORTANT)
    normal_df = df[df['Class'] == 0]
    fraud_df = df[df['Class'] == 1]

    train_df, test_normal_df = train_test_split(
        normal_df, test_size=0.2, random_state=42
    )

    test_df = pd.concat([test_normal_df, fraud_df])

    print("\n📦 Data Split Completed:")
    print("Train Shape (ONLY NORMAL):", train_df.shape)
    print("Test Shape (Normal + Fraud):", test_df.shape)

    # Save for later phases
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\n💾 Train & Test datasets saved in /data folder")



if __name__ == "__main__":
    main()
    train_models()