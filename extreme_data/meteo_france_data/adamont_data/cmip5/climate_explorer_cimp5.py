import calendar
import os.path as op
import pandas as pd
import subprocess
from datetime import datetime, timedelta

import cdsapi
import numpy as np
from netCDF4._netCDF4 import Dataset, OrderedDict

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_rnumber, get_gcm_list
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_year_min_and_year_max_from_scenario, \
    AdamontScenario, adamont_scenarios_real
from extreme_data.utils import DATA_PATH

GLOBALTEMP_WEB_PATH = "https://climexp.knmi.nl/CMIP5/Tglobal/"
GLOBALTEMP_DATA_PATH = op.join(DATA_PATH, 'CMIP5_global_temp')


def get_scenario_name(scenario):
    if scenario is AdamontScenario.histo:
        return 'historicalNat'
    else:
        return str(scenario).split('.')[-1]


def year_to_global_mean_temp(gcm, scenario):
    # Compute everything
    ensemble_member = 'r{}i1p1'.format(gcm_to_rnumber[gcm])
    scenario_name = get_scenario_name(scenario)

    # Standards
    mean_annual_column_name = 'Annual mean'
    filename = 'global_tas_Amon_{}_{}_{}'.format(gcm, scenario_name, ensemble_member)
    dat_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.dat')
    txt_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.txt')
    csv_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.csv')
    # Download if needed
    if not op.exists(txt_filepath):
        download_dat(dat_filepath, txt_filepath)
    # Transform nc file into csv file
    if not op.exists(csv_filepath):
        dat_to_csv(csv_filepath, txt_filepath, mean_annual_column_name)

    # Load csv file
    df = pd.read_csv(csv_filepath, index_col=0)
    d = OrderedDict(df[mean_annual_column_name])
    print(gcm, scenario_name, np.mean(list(d.values())))
    return d


def dat_to_csv(csv_filepath, txt_filepath, mean_annual_column_name):
    d = OrderedDict()
    with open(txt_filepath, 'r') as f:
        for i, l in enumerate(f):
            year, l = l[:8], l[8:]
            month_temp = [float(f) for f in l.split()]
            assert len(month_temp) == 12
            d[year] = list(month_temp)
    df = pd.DataFrame.from_dict(d)
    df = df.transpose()
    df.columns = list(calendar.month_abbr)[1:]
    df[mean_annual_column_name] = df.mean(axis=1)
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
            year_to_global_temp = year_to_global_mean_temp(gcm, scenario)
            print(year_to_global_temp)


if __name__ == '__main__':
    # main_example()
    main_test_cmip5_loader()
