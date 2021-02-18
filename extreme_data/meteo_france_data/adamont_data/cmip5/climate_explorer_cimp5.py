import calendar
import os.path as op
import pandas as pd
import subprocess
from datetime import datetime, timedelta

import cdsapi
import numpy as np
from netCDF4._netCDF4 import Dataset, OrderedDict

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_rnumber
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_year_min_and_year_max_from_scenario, \
    AdamontScenario, adamont_scenarios_real, get_gcm_list
from extreme_data.utils import DATA_PATH

GLOBALTEMP_WEB_PATH = "https://climexp.knmi.nl/CMIP5/Tglobal/"
GLOBALTEMP_DATA_PATH = op.join(DATA_PATH, 'CMIP5_global_temp')


def get_scenario_name(scenario):
    if scenario is AdamontScenario.histo:
        return 'historicalNat'
    else:
        return str(scenario).split('.')[-1]


def year_to_global_mean_temp(gcm, scenario, year_min=None, year_max=None, rolling=30, anomaly=False):
    d = OrderedDict()
    years, global_mean_temps = years_and_global_mean_temps(gcm, scenario, year_min, year_max, rolling=rolling, anomaly=anomaly)
    for year, global_mean_temp in zip(years, global_mean_temps):
        d[year] = global_mean_temp
    return d


def years_and_global_mean_temps(gcm, scenario, year_min=None, year_max=None, rolling=30, anomaly=False):
    # Compute everything
    ensemble_member = 'r{}i1p1'.format(gcm_to_rnumber[gcm])
    scenario_name = get_scenario_name(scenario)

    # Standards
    mean_annual_column_name = 'Annual mean'
    anomaly_annual_column_name = 'Annual anomaly'
    rolling_mean_annual_column_name = 'Rolling annual mean for window={}'.format(rolling)
    rolling_anomaly_annual_column_name = 'Rolling annual anomaly for window={}'.format(rolling)
    filename = 'global_tas_Amon_{}_{}_{}'.format(gcm, scenario_name, ensemble_member)
    dat_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.dat')
    txt_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.txt')
    csv_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.csv')
    # Download if needed
    if not op.exists(txt_filepath):
        download_dat(dat_filepath, txt_filepath)
    # Transform nc file into csv file
    if not op.exists(csv_filepath):
        dat_to_csv(csv_filepath, txt_filepath, mean_annual_column_name, rolling_mean_annual_column_name,
                   anomaly_annual_column_name,  rolling_anomaly_annual_column_name,
                   rolling=rolling)

    # Load csv file
    df = pd.read_csv(csv_filepath, index_col=0)
    if year_min is None:
        year_min = df.index[0]
    if year_max is None:
        year_max = df.index[-1]
    df = df.loc[year_min:year_max]
    years = list(df.index)
    assert years[0] >= year_min
    assert years[-1] <= year_max
    if rolling:
        if anomaly:
            global_mean_temp = list(df[rolling_anomaly_annual_column_name])
        else:
            global_mean_temp = list(df[rolling_mean_annual_column_name])
    else:
        if anomaly:
            global_mean_temp = list(df[anomaly_annual_column_name])
        else:
            global_mean_temp = list(df[mean_annual_column_name])
    return years, global_mean_temp


def dat_to_csv(csv_filepath, txt_filepath, mean_annual_column_name, rolling_mean_annual_column_name,
               anomaly_annual_column_name,  rolling_anomaly_annual_column_name, rolling=30):
    d = OrderedDict()
    with open(txt_filepath, 'r') as f:
        for i, l in enumerate(f):
            year, l = int(l[:5]), l[8:]
            month_temp = [float(f) for f in l.split()]
            assert len(month_temp) == 12
            d[int(year)] = list(month_temp)
    df = pd.DataFrame.from_dict(d)
    df = df.transpose()
    df.columns = list(calendar.month_abbr)[1:]
    df_temp_until_july = df.iloc[1:, :7]
    assert len(df_temp_until_july.columns) == 7
    df_temp_after_august = df.iloc[:-1, 7:]
    assert len(df_temp_after_august.columns) == 5
    l = df_temp_until_july.sum(axis=1).values + df_temp_after_august.sum(axis=1).values
    l /= 12
    l = [np.nan] + list(l)
    assert len(l) == len(df.index)
    df[mean_annual_column_name] = l
    s_mean_for_reference_period_1850_to_1900 = df.loc[1850:1900, mean_annual_column_name]
    # Sometimes some initial global mean temperatures are negative for the first years,
    # we remove them for the computation of the mean
    ind = s_mean_for_reference_period_1850_to_1900 > 0
    mean_for_reference_period_1850_to_1900 = s_mean_for_reference_period_1850_to_1900.loc[ind].mean()
    df[anomaly_annual_column_name] = df[mean_annual_column_name] - mean_for_reference_period_1850_to_1900
    # Computing the rolling
    if rolling is not None:
        df[rolling_mean_annual_column_name] = df[mean_annual_column_name].rolling(window=rolling).mean()
        df[rolling_anomaly_annual_column_name] = df[anomaly_annual_column_name].rolling(window=rolling).mean()
    df.to_csv(csv_filepath)


def download_dat(dat_filepath, txt_filepath):
    web_filepath = op.join(GLOBALTEMP_WEB_PATH, op.basename(dat_filepath))
    dirname = op.dirname(dat_filepath)
    requests = [
        'wget {} -P {}'.format(web_filepath, dirname),
        'tail -n +4 {} > {}'.format(dat_filepath, txt_filepath),
    ]
    for request in requests:
        subprocess.run(request, shell=True)


def main_example():
    scenario = AdamontScenario.rcp45
    gcm = 'EC-EARTH'
    year_to_global_mean_temp(gcm, scenario)


def main_test_cmip5_loader():
    for scenario in adamont_scenarios_real[1:]:
        for gcm in get_gcm_list(adamont_version=2)[:]:
            print(gcm, scenario)
            years, temps = years_and_global_mean_temps(gcm, scenario)
            print(years)
            print(temps)


def test_rolling():
    df = pd.DataFrame([1, 2, 3, 4, 5])
    print(df)
    df2 = df.rolling(window=3).mean()
    print(df2)


if __name__ == '__main__':
    # main_example()
    # test_rolling()
    main_test_cmip5_loader()
