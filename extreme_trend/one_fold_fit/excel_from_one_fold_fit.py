import os
import os.path as op
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from extreme_data.cru_data.global_mean_temperature_until_2020 import _year_to_average_global_mean_temp
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import get_closest_year, \
    year_to_averaged_global_mean_temp, years_and_global_mean_temps, year_to_global_mean_temp
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.utils import RESULTS_PATH
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from root_utils import get_display_name_from_object_type, SHORT_VERSION_TIME


#  Excel writing
def to_excel(one_fold_fit, gcm_rcm_couple_to_studies):
    gcm_rcm_couples = list(gcm_rcm_couple_to_studies.keys())
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
    df_temperature_sheet(one_fold_fit).to_excel(writer, "quantile(temps)", index=False)
    df_temperature_gcm_sheet(gcm_rcm_couples).to_excel(writer, "rechauffement pour chaque gcm", index=False)
    df_maxima(one_fold_fit, gcm_rcm_couple_to_studies).to_excel(writer, "maxima pour chaque gcm-rcm", index=False)
    # Save and close writer
    writer.save()
    writer.close()

def df_temperature_sheet(one_fold_fit: OneFoldFit) -> pd.DataFrame:
    step = 0.1
    covariates = np.arange(1, 4 + step, step)
    return compute_df('GMST', one_fold_fit, covariates, covariates)

def df_temporal_sheet(one_fold_fit: OneFoldFit) -> pd.DataFrame:
    covariates_for_df = list(range(1950, 2101))
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

def get_year_to_obs_average():
    years, average = _year_to_average_global_mean_temp()
    return dict(zip(years, average))

def df_temperature_gcm_sheet(gcm_rcm_couples) -> pd.DataFrame:
    df = pd.DataFrame()
    covariates_for_df = list(range(1950, 2101))
    df['Years'] = covariates_for_df
    gcms = set([gcm for gcm, _ in gcm_rcm_couples])
    for gcm in gcms:
        if gcm is None:
            #  Obs case
            key = 'Observations'
            year_to_obs_average = get_year_to_obs_average()
            global_mean = [year_to_obs_average[year] if year in year_to_obs_average else np.nan for year in covariates_for_df]
        else:
            # Gcm case
            key = gcm
            d = year_to_global_mean_temp(gcm, AdamontScenario.rcp85_extended)
            global_mean = [d[year] for year in covariates_for_df]
        df[key] = global_mean
    return df

def df_maxima(one_fold_fit: OneFoldFit, gcm_rcm_couple_to_studies: Dict[Tuple[str, str], AltitudesStudies]) -> pd.DataFrame:
    df = pd.DataFrame()
    covariates_for_df = list(range(1950, 2101))
    df['Years'] = covariates_for_df
    for gcm_rcm_couple, studies in gcm_rcm_couple_to_studies.items():
        key = 'Observations' if gcm_rcm_couple[0] is None else '-'.join(list(gcm_rcm_couple))
        dataset = studies.spatio_temporal_dataset(one_fold_fit.massif_name, [one_fold_fit.altitude_plot])
        years = dataset.df_coordinates.iloc[:, 0].values
        maxima = dataset.maxima_gev.flatten()
        assert len(years) == len(maxima)
        year_to_maximum = dict(zip(years, maxima))
        global_mean = [year_to_maximum[year] if year in year_to_maximum else np.nan for year in covariates_for_df]
        df[key] = global_mean
    return df