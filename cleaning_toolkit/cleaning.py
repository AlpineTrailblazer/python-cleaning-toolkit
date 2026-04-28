import numpy as np
import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names by:
    - converting to lowercase
    - stripping leading and trailing whitespace
    - replacing spaces with underscores

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """

    # Create a copy to avoid modifying original DataFrame
    df = df.copy()

    # Apply transformations to column names
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


def fill_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str = 'mean'
) -> pd.DataFrame:
    """
    Fill missing values in a selected column using a chosen strategy.

    Parameters:
        df pd.DataFrame: Input DataFrame
        column (str): Name of the column to fill.
        strategy (str): Missing value fill strategy, three options-
            1. 'mean'
            2. 'median'
            3. 'mode'

    Returns:
        pd.DataFrame: DataFrame with missing values filled in the selected column.

    Raises:
        KeyError: If the column does not exist.
        ValueError: If an unsupported strategy is indicated.
    """

    df = df.copy()

    # Confirm selected colum exists before processing
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Calculate fill value based on selected strategy
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0]
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'mode'.")
    
    # Replace missing values in the selected column
    df[column] = df[column].fillna(fill_value)

    return df


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    errors: str = 'raise',
    rounding: str | None = None
) -> pd.DataFrame:
    """
    Convert a selected column to a specified data type.

    Parameters:
        df pd.DataFrame: Input DataFrame.
        column (str): Name of the column to convert.
        target_type (str): Target data type, four options-
            1. 'int'
            2. 'float'
            3. 'str'
            4. 'bool'
        errors (str): Error handling strategy, two options-
            1. 'raise' : stop on invalid values
            2. 'coerce' : convert invalid values to NaN
        rounding (str):  Optional rounding strategy for integer conversion, four options-
            1. None : strict, no decimal-to-int conversion
            2. 'round' : round to nearest whole number
            3. 'floor' : round down 
            4. 'ceil' : round up

    Returns:
        pd.DataFrame: DataFrame with the selected column converted.

    Raises:
        KeyError: If the column does not exist.
        ValueError: If an unsupported target_type or errors option is indicated.
    """
    df = df.copy()

    # Confirm selected colum exists before processing
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    # Confirm error handling option is supported
    if errors not in ['raise', 'coerce']:
        raise ValueError("Errors must be either 'raise' or 'coerce'.")
    
    # Confirm rounding option is supported
    if rounding not in [None, 'round', 'floor', 'ceil']:
        raise ValueError("Rounding must be None, 'round', 'floor', or 'ceil'.")

    # Convert column based on selected target type
    if target_type == 'int':
        numeric_values = pd.to_numeric(df[column], errors = errors)

        if rounding == 'round':
            numeric_values = numeric_values.round()
        elif rounding == 'floor':
            numeric_values = numeric_values.apply(np.floor)
        elif rounding == 'ceil':
            numeric_values = numeric_values.apply(np.ceil)

        df[column] = numeric_values.astype('Int64')

    elif target_type == 'float':
        df[column] = pd.to_numeric(df[column], errors = errors).astype(float)
    elif target_type == 'str':
        df[column] = df[column].astype(str)
    elif target_type == 'bool':
        bool_map = {
            'true': True,
            'false': False,
            'yes': True,
            'no': False,
            'y': True,
            'n': False,
            '1': True,
            '0': False,
        }

        # Standardize values before boolean mapping
        normalized_values = df[column].astype(str).str.strip().str.lower()

        # Identify values that cannot be mapped to boolean values
        invalid_values = normalized_values[
            ~normalized_values.isin(bool_map.keys()) & df[column].notna()
        ]

        if not invalid_values.empty and errors == "raise":
            raise ValueError(
                f"Column '{column}' contains values that cannot be converted to bool: "
                f"{invalid_values.unique().tolist()}"
            )
        
        df[column] = normalized_values.map(bool_map).astype("boolean")
    else:
        raise ValueError("The target_type must be 'int', 'float', 'str', or 'bool'.")
    
    return df