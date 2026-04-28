import pandas as pd

from cleaning_toolkit.cleaning import standardize_columns, fill_missing, convert_column_type
from cleaning_toolkit.encoding import one_hot_encode
from cleaning_toolkit.validation import check_missing, validate_column_exists
from cleaning_toolkit.pipeline import clean_dataframe


# Test DataFrame creation
data = {
    "Pet_Name": ['Zulu', 'Nyx', 'Roxy', 'Cider', 'Nanook'],
    "Pet AGE": [8, None, 2, 2, 1],
    "Pet Feed in Cups ": [4, 1.5, None, 2.5, 4],
    "Takes RX": ['yes', 'yes', 'no', 'no', 'no']
}

df = pd.DataFrame(data)

print("Original:")
print(df)

# Test standardize_columns function
df_clean = standardize_columns(df)

print("\nStandardized columns:")
print(df_clean)

# Test fill_missing function
df_filled = fill_missing(df_clean, 'pet_feed_in_cups', strategy='mean')

print("\nMissing values filled:")
print(df_filled)

# Test convert_column_type function
df_typed = convert_column_type(
    df_filled, 
    column='pet_age',
    target_type='int',
    errors='raise',
    rounding='round'
)

print("\nConverted pet_age to integer:")
print(df_typed)
print(df_typed.dtypes)

# Validation test
print("\nMissing value counts:")
print(check_missing(df_typed))

validate_column_exists(df_typed, 'pet_age')

# Test one_hot_encode function
df_encoded = one_hot_encode(df_typed, column='takes_rx')

print("\nOne-hot encoded column:")
print(df_encoded)

# Test master pipeline
df_master = clean_dataframe(
    df,
    missing_strategies = {'pet_age': 'mean', 'pet_feed_in_cups': 'median'},
    type_conversions = {
        'pet_age': {
            'target_type': 'int',
            'errors': 'raise',
            'rounding': 'round'
        },
        'pet_feed_in_cups': {
            'target_type': 'float'
        }
    },
    encode_columns = ['takes_rx']
)

print("\nPipeline cleaned DataFrame:")
print(df_master)
print(df_master.dtypes)