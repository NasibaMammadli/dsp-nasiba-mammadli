"""Preprocessing functions for the house prices prediction model."""

from typing import Tuple, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target variable.
    
    Args:
        df: Input dataframe containing both features and target
        
    Returns:
        Tuple containing features dataframe and target series
    """
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    return X, y


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple containing lists of numeric and categorical column names
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with missing values handled
    """
    df_processed = df.copy()
    
    # Handle numeric missing values with median
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(
            df_processed[col].median()
        )
    
    # Handle categorical missing values with mode
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(
            df_processed[col].mode()[0]
        )
    
    return df_processed


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical features using OneHotEncoder.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        encoder: Optional pre-fitted OneHotEncoder
        
    Returns:
        Tuple of (encoded dataframe, fitted encoder)
    """
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_cols])
    else:
        encoded_data = encoder.transform(df[categorical_cols])
    
    # Create feature names
    feature_names = []
    for i, col in enumerate(categorical_cols):
        feature_names.extend(
            [f"{col}_{val}" for val in encoder.categories_[i]]
        )
    
    # Create encoded dataframe
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=feature_names,
        index=df.index
    )
    
    # Drop original categorical columns
    df_encoded = df.drop(columns=categorical_cols)
    
    # Concatenate encoded features
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded, encoder


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        scaler: Optional pre-fitted StandardScaler
        
    Returns:
        Tuple of (scaled dataframe, fitted scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
    else:
        scaled_data = scaler.transform(df[numeric_cols])
    
    # Create scaled dataframe
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=numeric_cols,
        index=df.index
    )
    
    # Drop original numeric columns
    df_scaled = df.drop(columns=numeric_cols)
    
    # Concatenate scaled features
    df_scaled = pd.concat([df_scaled, scaled_df], axis=1)
    
    return df_scaled, scaler
