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

# Filter the DataFrame to include only rows with LocationAbbr in the list of US state abbreviations
df_us_states = df[df['LocationAbbr'].isin(us_states)]


# %%
df_filtered = df[(df['DataValueAlt'] != df['DataValue']) & df['DataValueAlt'].notna() & df['DataValue'].notna()]
df_filtered[['DataValueAlt', 'DataValue']]


# %%
mappings = {
    'Topic': 'TopicID',
    'Question': 'QuestionID',
    'DataValueType': 'DataValueTypeID',
    'StratificationCategory1': 'StratificationCategoryID1',
    'Stratification1': 'StratificationID1',
    'LocationDesc': 'LocationAbbr'
}

# Iterate over each mapping and save to a new CSV file
for column, id_column in mappings.items():
    # Select the columns
    mapping_df = df[[column, id_column]]
    
    # Drop duplicate rows to ensure one-to-one mapping
    mapping_df = mapping_df.drop_duplicates()
    
    # Save the mapping relation to a new CSV file
    mapping_df.to_csv(f'{column.lower()}_mapping.csv', index=False)
    
    # Print a confirmation message
    print(f"Mapping relation for '{column}' saved to '{column.lower()}_mapping.csv'")
# %%
