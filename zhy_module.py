# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')

# %%
us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 
             'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 
             'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 
             'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
             'VA', 'WA', 'WV', 'WI', 'WY']

df_us_states = df[df['LocationAbbr'].isin(us_states)]

# %%
df_filtered = df[(df['DataValueAlt'] != df['DataValue']) & df['DataValueAlt'].notna() & df['DataValue'].notna()]
df_filtered[['DataValueAlt', 'DataValue']]

# %%
topic_mapping = df[['Topic', 'TopicID']].drop_duplicates().set_index('Topic').to_dict()['TopicID']
question_mapping = df[['Question', 'QuestionID']].drop_duplicates().set_index('Question').to_dict()['QuestionID']
datavalue_type_mapping = df[['DataValueType', 'DataValueTypeID']].drop_duplicates().set_index('DataValueType').to_dict()['DataValueTypeID']
stratification_category_mapping = df[['StratificationCategory1', 'StratificationCategoryID1']].drop_duplicates().set_index('StratificationCategory1').to_dict()['StratificationCategoryID1']
stratification_mapping = df[['Stratification1', 'StratificationID1']].drop_duplicates().set_index('Stratification1').to_dict()['StratificationID1']
location_mapping = df[['LocationDesc', 'LocationAbbr']].drop_duplicates().set_index('LocationDesc').to_dict()['LocationAbbr']

# %%

# %%
