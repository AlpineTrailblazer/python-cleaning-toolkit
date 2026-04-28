import pandas as pd


def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    One-hot encode a categorical column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the categorical column to encode.
        
    Returns:
        pd.DataFrame: DataFrame with the selected column one-hot encoded.
    """
    df = df.copy()

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    encoded_df = pd.get_dummies(df, columns=[column], dtype=int)

    return encoded_df