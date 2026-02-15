import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

#----------------------------------------------------------------------------------------------------
# RELOAD MODULE
#----------------------------------------------------------------------------------------------------
def reload_module():
    pass


#----------------------------------------------------------------------------------------------------
# CLEANING
#----------------------------------------------------------------------------------------------------

import unicodedata

CYRILLIC_TO_LATIN = {
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X",
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x"
}

def clean_string(s):

    if isinstance(s, str):

        # Normalize Unicode
        s = unicodedata.normalize("NFKD", s)

        # Replace Cyrillic lookalikes with Latin equivalents
        for cyr, lat in CYRILLIC_TO_LATIN.items():
            s = s.replace(cyr, lat)

        # Remove accents
        s = s.encode("ascii", "ignore").decode("ascii")

        # Replace weird spaces/dashes
        s = s.replace("\xa0", " ").replace("–", "-").replace("—", "-")

        # Strip whitespace
        s = s.strip()

    return s


def clean_string_old(s):
    """
    Clean a value by normalizing Unicode, removing non‑ASCII characters,
    fixing common formatting issues, and trimming whitespace.
    """

    if isinstance(s, str):

        # Normalize Unicode (NFKD breaks characters into base + accents)
        s = unicodedata.normalize("NFKD", s)

        # Remove accents and weird diacritics
        s = s.encode("ascii", "ignore").decode("ascii")

        # Replace weird spaces/dashes
        s = s.replace("\xa0", " ").replace("–", "-").replace("—", "-")

        # Strip extra whitespace
        s = s.strip()

    return s

#----------------------------------------------------------------------------------------------------

def find_decimal_records(df, column, extra_cols=None, regex=r'\.\d+'):
    """
    Identify rows where a column contains decimal values (e.g., '123.45').

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to search.
    column : str
        The column to check for decimal patterns.
    extra_cols : list of str, optional
        Additional columns to return along with the matched rows.
    regex : str, optional
        Regex pattern to detect decimals. Default = r'\.\d+'.

    Returns
    -------
    pandas.DataFrame
        Subset of df containing rows where the column matches the regex.
    """

    # Build mask for decimal detection
    mask = df[column].astype(str).str.contains(regex, na=False)

    # Columns to return
    cols_to_show = [column]
    if extra_cols:
        cols_to_show += extra_cols

    return df.loc[mask, cols_to_show]

def find_nonzero_decimals(df, column, extra_cols=None):
    """
    Return rows where `column` contains a decimal value with a non-zero decimal part.
    Example matches: 12.5, 3.01, 7.10
    Non-matches: 12.0, 7.00, 5, 3.
    """

    # Regex: decimal point + digits + at least one non-zero digit
    pattern = r'\.\d*[1-9]\d*'

    # Build mask
    mask = df[column].astype(str).str.contains(pattern, na=False)

    # Columns to return
    cols_to_show = [column]
    if extra_cols:
        cols_to_show += extra_cols

    result = df.loc[mask, cols_to_show]

    # If no matches, notify user
    if result.empty:
        print(f"No non-zero decimal values found in column '{column}'.")
        return result

    return result

#----------------------------------------------------------------------------------------------------

semantic_to_dtype = {
    'numeric discrete': 'float64',            
    'numeric continuous': 'float64',
    'categorical nominal': 'category',
    'categorical ordinal': 'category',      # we can add ordered categories later
}

def build_dtype_mapping(df, data_dict, semantic_to_dtype):
    """
    Build a mapping of dataframe columns to target pandas dtypes
    based on semantic types in the data dictionary.
    """
    semantic = data_dict['Semantic Type']

    col_to_dtype = {
        col: semantic_to_dtype.get(semantic.get(col))
        for col in df.columns
        if pd.notna(semantic.get(col)) 
        and semantic_to_dtype.get(semantic.get(col)) is not None
    }

    return col_to_dtype

def convert_column_dtype(df, col, dtype):
    """
    Convert a single column to the specified dtype.
    """
    if dtype == 'Int64':
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    elif dtype == 'float64':
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    elif dtype == 'category':
        df[col] = df[col].astype('category')

    return df

def apply_dtypes_from_mapping(df, col_to_dtype):
    """
    Apply dtype conversions to all columns in the mapping.
    """
    for col, dtype in col_to_dtype.items():
        df = convert_column_dtype(df, col, dtype)

    return df

def apply_semantic_dtypes(df, data_dict, semantic_to_dtype):
    """
    Orchestrator: build mapping → apply conversions → return updated df.
    """
    # Step 1: Build mapping
    col_to_dtype = build_dtype_mapping(df, data_dict, semantic_to_dtype)

    # Step 2: Apply conversions
    df = apply_dtypes_from_mapping(df, col_to_dtype)

    return df