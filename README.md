# Python Cleaning Toolkit
 A reusable Python toolkit for basic data preprocessing workflows, including column standardization, missing value handling, type conversion, validation, categorical encoding, and pipeline-based cleaning.

## Project Goal
 Build a modular, production-style cleaning pipeline that I can reuse preparing tabular data for analysis or machine learning.

## Features
 - Standardize column names
 - Fill missing values using mean, median, or mode
 - Convert column data types with configurable error handling
 - One-hot encode categorical columns
 - Check missing value counts
 - Validate column existence
 - Run a master cleaning pipeline


## Tech Stack
 - Python
 - pandas
 - NumPy


## How to Use
 1. *Install dependencies* 
        pip install -r requirements.txt
 2. *Import the pipeline*
        from cleaning_toolkit.pipeline import clean_dataframe
 3. *Load your dataset into a pandas DataFrame*
        df = pd.read_csv('your_dataset.csv')
 4. *Apply the cleaning pipeline*
        cleaned_df = clean_dataframe(
            df,
            missing_strategies={
                'age': 'mean',
                'income': 'median',
            },
            type_conversions={
                'age': {
                    'target_type': 'int',
                    'rounding': 'round',
                },
                'income': {
                    'target_type': 'float',
                }
            },
            encode_columns=['category']
        )
 5. *Inspect the cleaned DataFrame*
        print(cleaned_df.head())
        print(cleaned_df.dtypes)

**Notes:**
 - Column names are automatically standardized (lowercase, no spaces)
 - Missing value handling must be defined per column
 - Integer conversion is strict by default- rounding must be explicitly stated
 - Encoding replaces categorical columns with one-hot encoded columns


## What I Learned
 - How to structure reusable Python modules
 - How to write small, single-purpose preprocessing functions
 - How to validate user inputs and raise clear errors
 - How to build a simple pipeline function from modular components
 - How to avoid silent data corruption during type conversion


## Future Improvements
 - Add automated tests with Pytest
 - Add support for multiple encoding strategies
 - Add string-normalization tools (case, punctuation, pattern matching)
 - Add logging
 - Turn the toolkit into a pip-installable package