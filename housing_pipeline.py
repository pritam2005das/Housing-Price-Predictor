import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_and_save_model(csv_path, pipeline_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Define categorical and numeric features
    categorical = [
        'furnishingstatus',
        'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    numeric = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    # Build preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', StandardScaler(), numeric)
        ]
    )

    # pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', LinearRegression())
    ])

    # Train
    pipeline.fit(X, y)

    # Save pipeline
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)

train_and_save_model('Housing.csv', 'housing_pipeline.pkl')