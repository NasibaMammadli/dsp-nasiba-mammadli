"""Inference functions for the house prices prediction model."""

from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib

from .preprocess import (
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features
)


def load_model_and_preprocessing() -> tuple[Any, Dict[str, Any]]:
    """Load the trained model and preprocessing objects.
    
    Returns:
        Tuple containing:
        - Loaded model
        - Dictionary containing preprocessing objects
    """
    model = joblib.load('models/model.joblib')
    preprocessing_objects = joblib.load('models/preprocessing.joblib')
    return model, preprocessing_objects


def prepare_inference_data(
    df: pd.DataFrame,
    preprocessing_objects: Dict[str, Any]
) -> pd.DataFrame:
    """Prepare data for inference using saved preprocessing objects.
    
    Args:
        df: Input dataframe
        preprocessing_objects: Dictionary containing preprocessing objects
        
    Returns:
        Processed dataframe ready for inference
    """
    # Get preprocessing objects
    numeric_cols = preprocessing_objects['numeric_cols']
    categorical_cols = preprocessing_objects['categorical_cols']
    encoder = preprocessing_objects['encoder']
    scaler = preprocessing_objects['scaler']
    
    # Handle missing values
    df = handle_missing_values(df, numeric_cols, categorical_cols)
    
    # Encode categorical features
    df, _ = encode_categorical_features(df, categorical_cols, encoder)
    
    # Scale numeric features
    df, _ = scale_numeric_features(df, numeric_cols, scaler)
    
    return df


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions using the trained model.
    
    Args:
        input_data: Input dataframe containing features
        
    Returns:
        Array of predicted house prices
    """
    # Load model and preprocessing objects
    model, preprocessing_objects = load_model_and_preprocessing()
    
    # Prepare data for inference
    processed_data = prepare_inference_data(input_data, preprocessing_objects)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    return predictions
