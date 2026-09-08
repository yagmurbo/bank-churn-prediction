import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, dtype=int)

def apply_new_features(df):
    df_new = df.copy()

    # dropping columns which are irrevelant

    df_new = df.drop(columns=['RowNumber', 'CustomerId', 'Complain', 'Surname'])

    # adding new features

    df_new['BalanceSalaryRatio'] = df_new['Balance'] / (df_new['EstimatedSalary'] + 1) # +1 is for avoiding division by zero
    df_new['TenureByAge'] = df_new['Tenure'] / (df_new['Age'] + 1)
    df_new['CreditScoreGivenAge'] = df_new['CreditScore'] / (df_new['Age'] + 1)

    # one hot encoding for categorical values

    categorical_cols = ['RowNumber', 'CustomerId', 'Surname']
    df_encoded = encoder.fit_transform(df_new[categorical_cols])
    df_new = pd.concat([df_new.drop(columns=categorical_cols), df_encoded], axis=1)

    return df_new