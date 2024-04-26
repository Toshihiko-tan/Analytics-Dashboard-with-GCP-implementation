# This module contains functions to clean and transform the data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_raw = pd.read_csv("U.S._Chronic_Disease_Indicators.csv")

# Drop columns that only contain na values
na_columns = ['Response', 'StratificationCategory2', 'Stratification2', 
    'StratificationCategory3', 'Stratification3', 'ResponseID', 
    'StratificationCategoryID2', 'StratificationID2', 
    'StratificationCategoryID3', 'StratificationID3']

# Drop columns that have an identifier
# We will be using the identifier to build the model
columns_with_id = ['Topic', 'Question', 'DataValueType', 'StratificationCategory1'
     , 'Stratification1']

# The columns about location contain similar information
# We will only keep one of them: 'LocationAbbr'
columns_with_location = ['LocationDesc', 'Geolocation', 'LocationID']
columns_to_drop = columns_with_id + columns_with_location + na_columns
df = df_raw.drop(columns=columns_to_drop, axis=1)

# Remove all the entries that have a footnote
df = df[df['DataValueFootnote'].isna()]
# Then, drop the two columns about footnote
df = df.drop(columns=['DataValueFootnote', 'DataValueFootnoteSymbol'])

# df.to_csv('cleaned_data.csv')

# %%
df_diabete = df[df['TopicID'] == 'DIA']
df_diabete.describe()
# %%
