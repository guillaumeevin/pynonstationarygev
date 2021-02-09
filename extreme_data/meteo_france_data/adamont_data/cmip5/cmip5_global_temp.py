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
        return 'historical'
    else:
        return str(scenario).split('.')[-1]


def get_year_min_and_year_max_for_global_temp(scenario):
    if scenario is AdamontScenario.histo:
        return 1951, 2005
    else:
        return 2006, 2100


def get_periods(gcm, scenario):
    if scenario is AdamontScenario.histo:
        if gcm == 'EC-EARTH':
            return ['195001-201212']
        else:
            return ['185001-200512']
    else:
        if gcm == 'CNRM-CM5':
            return []
        else:
            return ['200601-210012']
    
    


def year_to_global_mean_temp(gcm, scenario):
    # Compute everything
    periods = get_periods(gcm, scenario)
    ensemble_member = 'r{}i1p1'.format(gcm_to_rnumber[gcm])
    scenario_name = get_scenario_name(scenario)
    year_min, year_max = get_year_min_and_year_max_for_global_temp(scenario)

    # Standards
    mean_annual_column_name = 'Annual mean'
    zip_filepath = op.join(GLOBALTEMP_DATA_PATH, 'download.zip')


    # Create a csv file for each period
    for period in periods:
        filename = 'tas_Amon_{}_{}_{}_{}'.format(gcm, scenario_name, ensemble_member, period)
        nc_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.nc')
        csv_filepath = op.join(GLOBALTEMP_DATA_PATH, filename + '.csv')
        # Download if needed
        if not op.exists(nc_filepath):
            download_nc(ensemble_member, gcm, period, scenario_name, zip_filepath)
        # Transform nc file into csv file
        if not op.exists(csv_filepath):
            nc_to_csv(csv_filepath, mean_annual_column_name, nc_filepath, year_max, year_min)

    # Concatenate all csv together into a single summary_csv_filepath


    # Load csv file
    df = pd.read_csv(csv_filepath, index_col=0)
    d = OrderedDict(df[mean_annual_column_name])
    print(gcm, scenario_name, np.mean(list(d.values())))
    return d


def nc_to_csv(csv_filepath, mean_annual_column_name, nc_filepath, year_max, year_min):
    dataset = Dataset(nc_filepath)
    tas_list = np.array(dataset.variables['tas'])
    tas_list = np.mean(tas_list, axis=1)
    tas_list = np.mean(tas_list, axis=1)
    # 'days since 1850-1-1 00:00:00'
    time_list = np.array(dataset.variables['time'])
    assert len(time_list) == len(tas_list)
    start = datetime(year=1850, month=1, day=1, hour=0, minute=0, second=0)
    date_list = [start + timedelta(days=time) for time in time_list]
    winter_year_list = [date.year if date.month < 8 else date.year + 1 for date in date_list]
    winter_year_to_tas_list = {winter_year: [] for winter_year in range(year_min, year_max + 1)}
    for winter_year, tas in zip(winter_year_list, tas_list):
        if year_min <= winter_year <= year_max:
            winter_year_to_tas_list[winter_year].append(tas)
    # we have monthly values
    for tas_list in winter_year_to_tas_list.values():
        assert len(tas_list) == 12
    winter_year_to_mean_tas = OrderedDict()
    for winter_year, t in winter_year_to_tas_list.items():
        winter_year_to_mean_tas[winter_year] = np.mean(t)
    s = pd.Series(winter_year_to_mean_tas)
    df = pd.DataFrame({mean_annual_column_name: s})
    df.to_csv(csv_filepath)


def download_nc(ensemble_member, gcm, period, scenario_name, zip_filepath):
    gcm_lower = '_'.join(gcm.lower().split('-'))
    c = cdsapi.Client()
    c.retrieve(
        'projections-cmip5-monthly-single-levels',
        {
            'ensemble_member': ensemble_member,
            'format': 'zip',
            'experiment': scenario_name,
            'variable': '2m_temperature',
            'model': gcm_lower,
            'period': period,
        },
        zip_filepath)
    # unzip and delete
    request_list = [
        'unzip {} -d {}'.format(zip_filepath, op.dirname(zip_filepath)),
        'rm {}'.format(zip_filepath)
    ]
    for request in request_list:
        print(request)
        subprocess.run(request, shell=True)


def  main_example():
    scenario = AdamontScenario.histo
    gcm = 'EC-EARTH'
    year_to_global_mean_temp(gcm, scenario)


def main_test_cmip5_loader():
    for scenario in adamont_scenarios_real[:1]:
        for gcm in get_gcm_list(adamont_version=2)[:]:
            if gcm != 'CNRM-CM5':
                print(gcm, scenario)
                year_to_global_temp = year_to_global_mean_temp(gcm, scenario)
                print(year_to_global_temp)


if __name__ == '__main__':
    # main_example()
    main_test_cmip5_loader()
