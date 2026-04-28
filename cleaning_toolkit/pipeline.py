import pandas as pd

from cleaning_toolkit.cleaning import (
    standardize_columns,
    fill_missing,
    convert_column_type
)
from cleaning_toolkit.encoding import one_hot_encode


def clean_dataframe(
    df: pd.DataFrame,
    missing_strategies: dict | None = None,
    type_conversions: dict | None = None,
    encode_columns: list | None = None
) -> pd.DataFrame:
    """
    Apply a standard data cleaning pipeline to a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missing_strategies (dict | None): Mapping of column names to fill strategies.
        type_conversions (dict | None): Mapping of column names to target data types.
        encode_columns (list | None): List of categorical columns to one-hot encode.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()

    # Standardize column names before applying column specific operations
    df = standardize_columns(df)

    # Fill missing values based on provided column strategies
    if missing_strategies:
        for column, strategy in missing_strategies.items():
            df = fill_missing(df, column, strategy)

    # Convert column data types based on provided target types
    if type_conversions:
        for column, options in type_conversions.items():
            df = convert_column_type(
                df,
                column=column,
                target_type=options['target_type'],
                errors=options.get('errors', 'raise'),
                rounding=options.get('rounding')
            )

    # One-hot encode selected categorical columns
    if encode_columns:
        for column in encode_columns:
            df = one_hot_encode(df, column)

    return df