from sklearn.preprocessing import RobustScaler

def scale_features(train_df, test_df):
    scaler = RobustScaler()

    features = train_df.columns.drop('Class')

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[features] = scaler.fit_transform(train_df[features])
    test_scaled[features] = scaler.transform(test_df[features])

    return train_scaled, test_scaled