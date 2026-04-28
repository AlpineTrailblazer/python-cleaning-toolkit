import pandas as pd

def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Count missing values in each DataFrame column.
    
    Parameters:
        df pd.DataFrame: Input DataFrame
        
    Returns:
        pd.Series: Missing value counts by column.
    """
    return df.isna().sum()


def validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """
    Validate that a column exists in the DataFrame.
    
    Parameters:
        df pd.DataFrame: Input DataFrame.
        column (str): Column name to validate.
        
    Raises:
        KeyError: If the column does not exist.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")