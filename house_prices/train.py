"""Training functions for the house prices prediction model."""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from .preprocess import (
    split_features_target,
    identify_column_types,
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features
)


def prepare_training_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Prepare data for training by splitting and preprocessing.
    
    Args:
        df: Input dataframe containing both features and target
        
    Returns:
        Tuple containing:
        - Processed features dataframe
        - Target series
        - Dictionary containing fitted preprocessing objects
    """
    # Split features and target
    X, y = split_features_target(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X_train)
    
    # Handle missing values
    X_train = handle_missing_values(X_train, numeric_cols, categorical_cols)
    X_test = handle_missing_values(X_test, numeric_cols, categorical_cols)
    
    # Encode categorical features
    X_train, encoder = encode_categorical_features(X_train, categorical_cols)
    X_test, _ = encode_categorical_features(X_test, categorical_cols, encoder)
    
    # Scale numeric features
    X_train, scaler = scale_numeric_features(X_train, numeric_cols)
    X_test, _ = scale_numeric_features(X_test, numeric_cols, scaler)
    
    # Save preprocessing objects
    preprocessing_objects = {
        'encoder': encoder,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    return X_train, y_train, preprocessing_objects


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestRegressor:
    """Train the Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate the model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2
    }


def save_model_and_preprocessing(
    model: RandomForestRegressor,
    preprocessing_objects: Dict[str, Any]
) -> None:
    """Save the trained model and preprocessing objects.
    
    Args:
        model: Trained model
        preprocessing_objects: Dictionary containing preprocessing objects
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/model.joblib')
    
    # Save preprocessing objects
    joblib.dump(preprocessing_objects, 'models/preprocessing.joblib')


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """Main function to build and evaluate the model.
    
    Args:
        data: Input dataframe containing both features and target
        
    Returns:
        Dictionary containing model performance metrics
    """
    # Prepare data
    X_train, y_train, preprocessing_objects = prepare_training_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save model and preprocessing objects
    save_model_and_preprocessing(model, preprocessing_objects)
    
    # Evaluate model
    X_test, y_test, _ = prepare_training_data(data)
    performance_metrics = evaluate_model(model, X_test, y_test)
    
    return performance_metrics
