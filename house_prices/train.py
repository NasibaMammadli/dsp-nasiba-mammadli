"""Training functions for the house prices prediction model."""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from house_prices.preprocess import (
    split_features_target,
    identify_column_types,
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features
)

# Get the absolute path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def prepare_training_data(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Prepare data for training by preprocessing.
    
    Args:
        df: Input dataframe containing both features and target
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple containing:
        - Processed training features
        - Processed test features
        - Dictionary containing fitted preprocessing objects
    """
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(df)
    
    # Handle missing values
    X_train_processed = handle_missing_values(
        X_train, numeric_cols, categorical_cols
    )
    X_test_processed = handle_missing_values(
        X_test, numeric_cols, categorical_cols
    )
    
    # Encode categorical features
    X_train_encoded, encoder = encode_categorical_features(
        X_train_processed, categorical_cols, is_fit=True
    )
    X_test_encoded, _ = encode_categorical_features(
        X_test_processed, categorical_cols, encoder=encoder
    )
    
    # Scale numeric features
    X_train_scaled, scaler = scale_numeric_features(
        X_train_encoded, numeric_cols
    )
    X_test_scaled, _ = scale_numeric_features(
        X_test_encoded, numeric_cols, scaler
    )
    
    # Save preprocessing objects
    preprocessing_objects = {
        'encoder': encoder,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    return X_train_scaled, X_test_scaled, preprocessing_objects


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
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(MODELS_DIR, 'model.joblib'))
    
    # Save encoder and scaler separately
    joblib.dump(preprocessing_objects['encoder'], os.path.join(MODELS_DIR, 'encoder.joblib'))
    joblib.dump(preprocessing_objects['scaler'], os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    # Save other preprocessing objects
    joblib.dump(
        {
            'numeric_cols': preprocessing_objects['numeric_cols'],
            'categorical_cols': preprocessing_objects['categorical_cols']
        },
        os.path.join(MODELS_DIR, 'preprocessing.joblib')
    )


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """Build and train the house prices prediction model.
    
    Args:
        data: Input dataframe containing both features and target
        
    Returns:
        Dictionary containing model performance metrics
    """
    # Split features and target
    X, y = split_features_target(data)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Prepare training data
    X_train_processed, X_test_processed, preprocessing_objects = prepare_training_data(
        X, X_train, X_test
    )
    
    # Train model
    model = train_model(X_train_processed, y_train)
    
    # Evaluate model
    performance = evaluate_model(model, X_test_processed, y_test)
    
    # Save model and preprocessing objects
    save_model_and_preprocessing(model, preprocessing_objects)
    
    return performance
