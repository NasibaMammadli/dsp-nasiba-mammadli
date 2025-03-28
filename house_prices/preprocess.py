"""Preprocessing functions for the house prices prediction model."""

from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    numeric_cols = (
        df.select_dtypes(include=['int64', 'float64'])
        .columns.tolist()
    )
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> pd.DataFrame:
    """Handle missing values in the dataframe.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Dataframe with handled missing values
    """
    df_copy = df.copy()
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    return df_copy


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoder: OneHotEncoder = None
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical features using OneHotEncoder.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        encoder: Optional pre-fitted OneHotEncoder
        
    Returns:
        Tuple containing encoded dataframe and fitted encoder
    """
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[categorical_cols])
    else:
        encoded_features = encoder.transform(df[categorical_cols])
    
    # Create feature names for encoded columns
    feature_names = []
    for i, col in enumerate(categorical_cols):
        feature_names.extend(
            [f"{col}_{cat}" for cat in encoder.categories_[i]]
        )
    
    # Create dataframe with encoded features
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=feature_names,
        index=df.index
    )
    
    # Drop original categorical columns and concatenate encoded features
    result_df = pd.concat(
        [df.drop(categorical_cols, axis=1), encoded_df],
        axis=1
    )
    
    return result_df, encoder


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    scaler: StandardScaler = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        scaler: Optional pre-fitted StandardScaler
        
    Returns:
        Tuple containing scaled dataframe and fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df, scaler
