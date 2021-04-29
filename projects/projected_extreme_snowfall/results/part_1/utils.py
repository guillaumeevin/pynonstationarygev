import pandas as pd
import numpy as np
import os.path as op


def load_csv(csv_filepath):
    return pd.read_csv(csv_filepath, index_col=0) if op.exists(csv_filepath) else pd.DataFrame()


def is_already_done(csv_filepath, combination_name, altitude, gcm_rcm_couple):
    column_name = load_column_name(altitude, gcm_rcm_couple)
    df = load_csv(csv_filepath)
    if (combination_name in df.index) and (column_name in df.columns):
        missing_value = np.isnan(df.loc[combination_name, column_name])
        return not missing_value
    else:
        return False


def update_csv(csv_filepath, combination_name, altitude, gcm_rcm_couple, value):
    # Check value
    assert len(value) == 1
    value = value[0]
    # Load csv
    column_name = load_column_name(altitude, gcm_rcm_couple)
    df = load_csv(csv_filepath)
    # Add value dynamically
    if combination_name not in df.index:
        if df.empty:
            df = pd.DataFrame({column_name: [value]}, index=[combination_name])
        else:
            nb_index = len(df.index)
            df.loc[combination_name] = df.iloc[nb_index - 1] * np.nan
    if column_name not in df.columns:
        df[column_name] = np.nan
    df.loc[combination_name, column_name] = value
    # Compute sum on the column without gaps
    sum_column_name = 'sum'
    if sum_column_name in df.columns:
        df.drop(columns=[sum_column_name], inplace=True)
    df[sum_column_name] = df.sum(axis=1)
    df.sort_values(by=sum_column_name, inplace=True)
    print(df.head())
    # save intermediate results
    df.to_csv(csv_filepath)


def load_column_name(altitude, gcm_rcm_couple):
    return str(altitude) + '_' + '_'.join([c.replace('-', '') for c in gcm_rcm_couple])
