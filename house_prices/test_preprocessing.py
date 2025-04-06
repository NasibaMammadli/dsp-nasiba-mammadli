"""Test file to verify preprocessing behavior hasn't changed."""

import pandas as pd
import os
from house_prices.preprocess import (
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features,
    identify_column_types
)

def test_preprocessing_behavior():
    """Test that preprocessing behavior hasn't changed."""
    # Load original processed data
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data'
    )
    original_df = pd.read_parquet(
        os.path.join(data_dir, 'processed_df.parquet')
    )
    
    # Load raw data
    raw_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(raw_df)
    
    # Remove target column from numeric columns
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    
    # Process data with new functions
    processed_df = handle_missing_values(
        raw_df, numeric_cols, categorical_cols
    )
    processed_df, _ = encode_categorical_features(
        processed_df, categorical_cols
    )
    processed_df, _ = scale_numeric_features(
        processed_df, numeric_cols
    )
    
    # Compare results
    pd.testing.assert_frame_equal(
        processed_df,
        original_df,
        check_dtype=False
    )
    
    print("Preprocessing behavior test passed!")

if __name__ == '__main__':
    test_preprocessing_behavior() 