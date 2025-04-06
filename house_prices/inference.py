"""Inference functions for the house prices prediction model."""

import pandas as pd
import numpy as np
import joblib
import os

from house_prices.preprocess import (
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features
)

# Get the absolute path to the models directory
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'models'
)

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions on new data using the trained model.
    
    Args:
        input_data: Input dataframe containing features
        
    Returns:
        Array of predicted house prices
    """
    # Load model and preprocessors
    model = joblib.load(os.path.join(MODELS_DIR, 'model.joblib'))
    encoder = joblib.load(os.path.join(MODELS_DIR, 'encoder.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    preprocessing = joblib.load(
        os.path.join(MODELS_DIR, 'preprocessing.joblib')
    )
    
    # Get column types from preprocessing
    numeric_cols = preprocessing['numeric_cols']
    categorical_cols = preprocessing['categorical_cols']
    
    # Handle missing values
    input_processed = handle_missing_values(
        input_data, numeric_cols, categorical_cols
    )
    
    # Encode categorical features
    input_encoded, _ = encode_categorical_features(
        input_processed, categorical_cols, encoder=encoder
    )
    
    # Scale numeric features
    input_scaled, _ = scale_numeric_features(
        input_encoded, numeric_cols, scaler=scaler
    )
    
    # Make predictions
    predictions = model.predict(input_scaled)
    
    return predictions
