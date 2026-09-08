import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, dtype=int)

def apply_feature_engineering(df):
    df_new = df.copy()

    # dropping columns which are irrevelant

    cols_to_drop = ['RowNumber', 'CustomerId', 'Complain', 'Surname']
    existing_cols = [col for col in cols_to_drop if col in df.columns]

    df_new = df_new.drop(columns=existing_cols)

    # adding new features

    df_new['BalanceSalaryRatio'] = df_new['Balance'] / (df_new['EstimatedSalary'] + 1) # +1 is for avoiding division by zero
    df_new['TenureByAge'] = df_new['Tenure'] / (df_new['Age'] + 1)
    df_new['CreditScoreGivenAge'] = df_new['CreditScore'] / (df_new['Age'] + 1)

    # one hot encoding for categorical values

    categorical_cols = ['Gender', 'Geography', 'Card Type']
    df_encoded = encoder.fit_transform(df_new[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded_pd = pd.DataFrame(df_encoded, columns=encoded_cols, index=df_new.index)

    df_new = df_new.drop(columns=categorical_cols)
    df_new = pd.concat([df_new, df_encoded_pd], axis=1)

    return df_new