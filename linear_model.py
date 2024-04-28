import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle



cleaned_data = pd.read_csv('Data/cleaned_data.csv')
dia01 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA01') 
                     & (cleaned_data['StratificationCategoryID1'] == 'OVERALL') 
                     & (cleaned_data['DataValueTypeID'] == 'CRDPREV') 
                     ][['LocationAbbr', 'DataValue', 'YearStart']]
dia02 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA02') 
                     & (cleaned_data['StratificationCategoryID1'] == 'OVERALL') 
                     & (cleaned_data['DataValueTypeID'] == 'CRDPREV') 
                     ][['LocationAbbr', 'DataValue', 'YearStart']]
dia03 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA03')
                     & (cleaned_data['StratificationCategoryID1'] == 'OVERALL')
                     & (cleaned_data['DataValueTypeID'] == 'CRDRATE') 
                     ][['LocationAbbr', 'DataValue', 'YearStart']]
dia04 = cleaned_data[(cleaned_data['QuestionID'] == 'DIA04')
                     & (cleaned_data['StratificationCategoryID1'] == 'OVERALL')
                     & (cleaned_data['DataValueTypeID'] == 'CRDRATE') 
                     ][['LocationAbbr', 'DataValue', 'YearStart']]
merged_df = pd.merge(dia01, dia02, on=['LocationAbbr', 'YearStart'], suffixes=('_dia01', '_dia02'))
merged_df = pd.merge(merged_df, dia03, on=['LocationAbbr', 'YearStart'])
merged_df = pd.merge(merged_df, dia04, on=['LocationAbbr', 'YearStart'], suffixes=('_dia03', '_dia04'))

# Drop the target variable 'DataValue_dia01' from x_train
x_train = merged_df.drop(['DataValue_dia01'], axis=1)

# Define columns to be one-hot encoded
columns_to_encode = ['LocationAbbr']

# Perform one-hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
x_train_encoded = ct.fit_transform(x_train)


# Fit a linear model
model = LinearRegression()
model.fit(x_train_encoded, merged_df['DataValue_dia01'])  # Using 'DataValue_dia01' as y_train for the model

# Save the model to a file
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(model, file)