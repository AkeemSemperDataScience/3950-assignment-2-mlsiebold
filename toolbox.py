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

def clean_string2(s):
    
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


def clean_string(s):
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
