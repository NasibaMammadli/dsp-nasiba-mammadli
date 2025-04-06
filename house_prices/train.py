"""Training functions for the house prices prediction model."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

from house_prices.preprocess import (
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features,
    identify_column_types
)


# Get the absolute path to the models directory
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'models'
)


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare training data by splitting and preprocessing.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(df)
    
    # Remove target column from numeric columns if present
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Split data into train and test sets
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Handle missing values
    X_train = handle_missing_values(X_train, numeric_cols, categorical_cols)
    X_test = handle_missing_values(X_test, numeric_cols, categorical_cols)
    
    # Encode categorical features
    X_train, encoder = encode_categorical_features(
        X_train, categorical_cols
    )
    X_test, _ = encode_categorical_features(
        X_test, categorical_cols, encoder=encoder
    )
    
    # Scale numeric features
    X_train, scaler = scale_numeric_features(X_train, numeric_cols)
    X_test, _ = scale_numeric_features(X_test, numeric_cols, scaler=scaler)
    
    # Save preprocessors
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoder, os.path.join(MODELS_DIR, 'encoder.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(
        {'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols},
        os.path.join(MODELS_DIR, 'preprocessing.joblib')
    )
    
    return X_train, X_test, y_train, y_test


def build_model(
    data: pd.DataFrame,
    target_col: str = 'SalePrice',
    test_size: float = 0.2,
    random_state: int = 42
) -> dict[str, float]:
    """Build and train the house prices prediction model.
    
    Args:
        data: Training data
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing model performance metrics
    """
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(
        data, target_col, test_size, random_state
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'model.joblib'))
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return {'rmse': rmse}
