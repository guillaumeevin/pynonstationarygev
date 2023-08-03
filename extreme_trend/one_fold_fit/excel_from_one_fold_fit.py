import os
import os.path as op

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import get_closest_year, \
    year_to_averaged_global_mean_temp
from extreme_data.utils import RESULTS_PATH
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from root_utils import get_display_name_from_object_type, SHORT_VERSION_TIME


#  Excel writing
def to_excel(one_fold_fit):
    #  Load writer
    model_name = get_display_name_from_object_type(one_fold_fit.models_classes[0])
    excel_filename = f'{one_fold_fit.massif_name}_{one_fold_fit.altitude_plot}_{model_name}.xlsx'
    path = op.join(RESULTS_PATH, SHORT_VERSION_TIME)
    if not op.exists(path):
        os.makedirs(path)
    excel_filepath = op.join(path, excel_filename)
    writer = pd.ExcelWriter(excel_filepath, engine='xlsxwriter')
    # Write sheetnames
    df_temperature_sheet(one_fold_fit).to_excel(writer, "quantile(rechauffement)", index=False)
    df_temporal_sheet(one_fold_fit).to_excel(writer, "quantile(temps)", index=False)
    # Save and close writer
    writer.save()
    writer.close()

def df_temperature_sheet(one_fold_fit: OneFoldFit) -> pd.DataFrame:
    step = 0.1
    covariates = np.arange(1, 4 + step, step)
    return compute_df('GMST', one_fold_fit, covariates, covariates)

def df_temporal_sheet(one_fold_fit: OneFoldFit) -> pd.DataFrame:
    covariates_for_df = list(range(2000, 2101))
    d = year_to_averaged_global_mean_temp(AdamontScenario.rcp85_extended, 1950, 2100)
    covariates = [d[year] for year in covariates_for_df]
    return compute_df('Year', one_fold_fit, covariates, covariates_for_df)

def compute_df(covariate_name, one_fold_fit, covariates, covariates_for_df):
    margin_function = one_fold_fit.best_estimator.margin_function_from_fit
    gev_params_list = []
    for covariate in covariates:
        gev_params = margin_function.get_params(np.array([covariate]))
        gev_params_list.append(gev_params)
    # Add information inside a dataframe
    df = pd.DataFrame()
    df[covariate_name] = covariates_for_df
    df['location'] = [gev_params.location for gev_params in gev_params_list]
    df['scale'] = [gev_params.scale for gev_params in gev_params_list]
    df['shape'] = [gev_params.shape for gev_params in gev_params_list]
    for return_period in [10, 100, 300]:
        df[f'{return_period}-year RL'] = [gev_params.return_level(return_period) for gev_params in gev_params_list]
    return df
