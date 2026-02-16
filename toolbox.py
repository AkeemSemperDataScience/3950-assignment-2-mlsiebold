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
# SET VARIABLES
#----------------------------------------------------------------------------------------------------

def groupby_unique(df, group_col, column):
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    if column not in df.columns:
        raise ValueError(f"Value column '{column}' not found in DataFrame.")

    return (
        df
        .groupby(group_col, observed=True)[column]
        .unique()
    )

#----------------------------------------------------------------------------------------------------

def detect_cat_cols(df):
    return df.select_dtypes(
        include=['object', 'category', 'string', 'bool']).columns.tolist()


def detect_num_cols(df):
    return df.select_dtypes(
        include=['number']).columns.tolist()

def detect_datetime_cols(df):
    return df.select_dtypes(
        include=['datetime64[ns]', 'datetimetz']).columns.tolist()

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
    """
    Normalize and sanitize a string by removing accents, replacing visually similar
    Cyrillic characters with Latin equivalents, fixing unusual whitespace/dash
    characters, and trimming surrounding spaces.

    This function performs the following steps:
    - Normalizes Unicode characters using NFKD form.
    - Replaces Cyrillic lookalike characters based on the CYRILLIC_TO_LATIN mapping.
    - Removes diacritics by encoding to ASCII and ignoring non‑ASCII characters.
    - Replaces non‑standard spaces and dash characters with standard equivalents.
    - Strips leading and trailing whitespace.
    - Converts string to lowercase

    Parameters
    ----------
    s : any
        The value to clean. Only string inputs are modified; all other types are
        returned unchanged.

    Returns
    -------
    any
        A cleaned string if `s` is a string, otherwise the original value.
    """

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

        # Convert to lowercase
        s = s.lower()

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

def clean_null_like(df, columns=None):
    null_like = {"<NA>", "nan", "none", "null", "na", "n/a", "", "missing", "?"}

    if columns is None:
        columns = df.select_dtypes(include=["object", "string", "category"]).columns

    for col in columns:
        df[col] = (
            df[col]
            .astype("string")            # safe: preserves <NA>
            .str.strip()
            .str.lower()
            .replace(null_like, pd.NA)
        )
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


#----------------------------------------------------------------------------------------------------
# EDA
#----------------------------------------------------------------------------------------------------

def eda_column_summary(df, column):
    """Perform EDA for a single column."""
    from pandas.api.types import is_numeric_dtype

    col = df[column]

    print(f"=== COLUMN: {column} ===")

    print("\nDTYPE:")
    print(col.dtype)

    print("\nNON-NULL VALUES:")
    print(f"{col.notnull().sum():,}")

    print("\nNULL VALUES:")
    print(f"{col.isnull().sum():,}")

    print("\nUNIQUE VALUES:")
    print(f"{col.nunique():,}")

    print("\nBASIC STATS:")
    print(col.describe())

    print("\nTOP VALUES:")
    print(col.value_counts().head(5))

    print("\nBOTTOM VALUES:")
    print(col.value_counts().tail(5))

def eda_dataset_summary(df):
    """Perform dataset-level EDA."""
    print("=== DATASET INFO ===")
    df.info()

    print("\n=== BASIC STATS ===")
    df.describe()

    print("\n=== NULL VALUES ===")
    print(df.null_values())

    print("\n=== DETECTED CATEGORICAL COLUMNS ===")
    print(df.detect_cat_cols())

    print("\n=== DETECTED NUMERIC COLUMNS ===")
    print(df.detect_num_cols())

    print("\n=== DATA TYPES ===")
    print(df.dtypes())

    print("\n=== TARGET ===")
    print(df.giveTarget())


#----------------------------------------------------------------------------------------------------
# PLOTTING
#----------------------------------------------------------------------------------------------------

def plot(df, columns):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col in detect_num_cols(df):
            plot_numeric(df, col)
        elif col in detect_cat_cols(df):
            plot_categorical(df, col)
        elif col in detect_datetime_cols(df):
            plot_datetime(df, col)
        else:
            print(f"Column '{col}' not recognized as numeric, categorical, or datetime.")

def plot_numeric(df, column):
    data = df[column].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {column}")

    sns.boxplot(x=data, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column}")

    sns.probplot(data, dist='norm', plot=axes[2])
    axes[2].set_title(f"Normal Q-Q Plot of {column}")

    plt.tight_layout()
    plt.show()            

def plot_categorical(df, column):
    data = df[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.countplot(x=data, ax=axes[0])
    axes[0].set_title(f"Countplot of {column}")

    sns.boxplot(x=data, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column}")

    axes[0].tick_params(axis='x', labelsize=6, rotation=90)
    axes[1].tick_params(axis='x', labelsize=6, rotation=90)

    plt.tight_layout()
    plt.show()    

def plot_datetime(df, column):
    data = df[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(x=data, ax=axes[0])
    axes[0].set_title(f"Time Series of {column}")

    sns.boxplot(x=data, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column}")

    axes[0].tick_params(axis='x', labelsize=6, rotation=90)
    axes[1].tick_params(axis='x', labelsize=6, rotation=90)

    plt.tight_layout()
    plt.show()

def plot_group_kde(df, groupby_col, columns):
    """
    Plot KDE curves for one or multiple numeric columns,
    grouped by the unique values in `group_col`.
    """

    # Allow a single column string
    if isinstance(columns, str):
        columns = [columns]

    # Validate group column
    if groupby_col not in df.columns:
        raise ValueError(f"Group column '{groupby_col}' not found in DataFrame.")

    # Get unique groups using your helper
    groups = df.set_unique_list(groupby_col)

    for col in columns:

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        if col not in df.detect_num_cols():
            raise TypeError(f"Column '{col}' must be numeric for KDE plotting.")

        plt.figure(figsize=(10, 5))

        # Plot KDE for each group
        for g in groups:
            subset = df[df[groupby_col] == g][col].dropna()
            if len(subset) > 1:  # KDE needs >1 data point
                sns.countplot(subset, label=str(g), fill=False)

        plt.title(f'Density by {groupby_col} for {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(title=groupby_col)

        plt.tight_layout()
        plt.show()

#----------------------------------------------------------------------------------------------------

def plot_group(df, groupby_col, columns):
    """
    Dispatch grouped plotting based on column type.
    """

    if isinstance(columns, str):
        columns = [columns]

    if groupby_col not in df.data.columns:
        raise ValueError(f"Group column '{groupby_col}' not found in DataFrame.")

    for col in columns:

        if col in df.detect_num_cols():
            df.plot_group_numeric(groupby_col, col)

        elif col in df.detect_cat_cols():
            df.plot_group_categorical(groupby_col, col)

        else:
            print(f"Column '{col}' not recognized as numeric or categorical.")

    # Grouped KDE for Numeric Columns

def plot_group_numeric(df, groupby_col, column):
    """
    Plot KDE curves for a numeric column grouped by group_col.
    """

    groups = df.set_unique_list(groupby_col)

    plt.figure(figsize=(10, 5))

    for g in groups:
        subset = df.data[df.data[groupby_col] == g][column].dropna()
        if len(subset) > 1:
            sns.kdeplot(subset, label=str(g), fill=False)

    plt.title(f"KDE by {groupby_col} for {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend(title=groupby_col)

    plt.tight_layout()
    plt.show()

def plot_group_categorical(df, groupby_col, column):
    """
    Plot a grouped countplot for a categorical column.
    """

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df.data,
        x=column,
        hue=groupby_col
    )

    plt.title(f"Countplot of {column} grouped by {groupby_col}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

