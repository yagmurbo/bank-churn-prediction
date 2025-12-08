# Dosya: src/features.py
import pandas as pd
import numpy as np

def apply_feature_engineering(df):

    #cleans data and removes data leakage, creates new features, encodes categorical variables

    df_new = df.copy()
    
    #data cleaning and removing leakage
    #Complain was leake feature so we drop it
    #RowNumber, CustomerId, Surname are identifiers and do not provide predictive power
    cols_to_drop = ['Complain', 'RowNumber', 'CustomerId', 'Surname']
    
    #only drop columns that exist in the dataframe
    existing_cols_to_drop = [c for c in cols_to_drop if c in df_new.columns]
    df_new = df_new.drop(existing_cols_to_drop, axis=1)
    
    #new feature creation
    
    #balance to salary ratio, tenure to age ratio, credit score to age ratio
    #to avoid division by zero, we add 1 to denominators if needed
    if 'EstimatedSalary' in df_new.columns and 'Balance' in df_new.columns:
        df_new['BalanceSalaryRatio'] = df_new['Balance'] / (df_new['EstimatedSalary'] + 1)
    
    #tenure/age ratio
    if 'Tenure' in df_new.columns and 'Age' in df_new.columns:
        df_new['TenureByAge'] = df_new['Tenure'] / (df_new['Age'] + 1)
    
    #credit score/age ratio
    if 'CreditScore' in df_new.columns and 'Age' in df_new.columns:
        df_new['CreditScoreGivenAge'] = df_new['CreditScore'] / (df_new['Age'] + 1)
    
    #categorical encoding
    #random forest can handle integer encoded categories, so we use simple mapping here
    
    #gender: female: 1, male: 0
    if 'Gender' in df_new.columns:
        df_new['Gender'] = df_new['Gender'].map({'Female': 1, 'Male': 0})
        #if there are any missing values after mapping, fill them with the mode (most frequent value)
        df_new['Gender'] = df_new['Gender'].fillna(0)
        
    #geography and card type: one-hot encoding

    categorical_cols = ['Geography', 'Card Type']

    existing_categorical = [c for c in categorical_cols if c in df_new.columns]
    if existing_categorical:
        df_new = pd.get_dummies(df_new, columns=existing_categorical, drop_first=True)

    return df_new