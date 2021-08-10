import pandas as pd
import numpy as np
import os.path as op

import xlrd

main_sheet_name = "Main"


def load_csv(csv_filepath):
    return pd.read_csv(csv_filepath, index_col=0) if op.exists(csv_filepath) else pd.DataFrame()


def load_excel(excel_filepath, sheetname):
    if not op.exists(excel_filepath):
        return pd.DataFrame()
    else:
        try:
            return pd.read_excel(excel_filepath, index_col=0, sheet_name=sheetname)
        except xlrd.biffh.XLRDError:
            return pd.DataFrame()


def is_already_done(excel_filepath, combination_name, altitude, gcm_rcm_couple):
    column_name = load_column_name(altitude, gcm_rcm_couple)
    df = load_excel(excel_filepath, main_sheet_name)
    if (combination_name in df.index) and (column_name in df.columns):
        missing_value = np.isnan(df.loc[combination_name, column_name])
        return not missing_value
    else:
        return False


def update_csv(excel_filepath, row_name, altitude, gcm_rcm_couple, value_list, sheetname=None):
    # Check value
    writer = pd.ExcelWriter(excel_filepath, engine='xlsxwriter')
    # Update main result
    column_name = load_column_name(altitude, gcm_rcm_couple)
    # update sub result
    for split in [None, "early", "later"][:1]:
        if sheetname is None:

            local_sheetname = main_sheet_name
            value = np.mean(value_list)
            if split is not None:
                local_sheetname += ' ' + split
                value = np.mean(value_list[:40]) if split == "early" else np.sum(value_list[-40:])
        else:
            local_sheetname = sheetname
            value = value_list

        df = load_excel(excel_filepath, local_sheetname)
        df = add_dynamical_value(column_name, row_name, df, value)
        # Compute sum on the column without gaps
        # sum_column_name = 'sum'
        mean_column_name = 'pourcentage of massif where approach better than without coef'
        # if sum_column_name in df.columns:
        #     df.drop(columns=[sum_column_name], inplace=True)
        if mean_column_name in df.columns:
            df.drop(columns=[mean_column_name], inplace=True)
        # df[sum_column_name] = df.sum(axis=1)
        df2 = df > df.loc["no effect"]
        df[mean_column_name] = df2.mean(axis=1) * 100
        df.sort_values(by=mean_column_name, inplace=True)
        # save intermediate results
        df.to_excel(writer, local_sheetname)
    # df2 = load_excel(excel_filepath, column_name)
    # years = list(range(2020, 2101))
    # for year, nllh in zip(years, value_list):
    #     df2 = add_dynamical_value(str(year), combination_name, df2, nllh)
    # df2.to_excel(writer, column_name)
    writer.save()
    writer.close()


def add_dynamical_value(column_name, row_name, df, value):
    # Add value dynamically
    if row_name not in df.index:
        if df.empty:
            df = pd.DataFrame({column_name: [value]}, index=[row_name])
        else:
            nb_index = len(df.index)
            df.loc[row_name] = df.iloc[nb_index - 1] * np.nan
    if column_name not in df.columns:
        df[column_name] = np.nan
    df.loc[row_name, column_name] = value
    return df


def load_column_name(altitude, gcm_rcm_couple):
    return str(altitude) + '_' + '_'.join([c.replace('-', '') for c in gcm_rcm_couple])
