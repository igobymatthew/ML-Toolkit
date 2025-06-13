import pandas as pd

def preprocess_data(df, target_column):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X)
    y = pd.factorize(y)[0]
    return X, y
