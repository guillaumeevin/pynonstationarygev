from collections import OrderedDict

import matplotlib as mpl
import pandas as pd
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall5Days, SafranSnowfall7Days, \
    SafranRainfall1Day, SafranRainfall3Days, \
    SafranRainfall5Days, SafranRainfall7Days, SafranRainfall, SafranSnowfall

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, \
    CrocusSnowLoad5Days, CrocusSnowLoad7Days, CrocusSnowLoad1Day, Crocus
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days


def compute_ratio(rainfall, snowfall):
    total_precipitation = rainfall + snowfall
    has_some_year_without_precipitation = (total_precipitation == 0).any()
    if has_some_year_without_precipitation:
        print('Nan values because we have some year without precipitation')
    ratio = snowfall / total_precipitation
    return ratio


def df_snow_load_maxima_parittion(rainfall_study: SafranRainfall, snowfall_study: SafranSnowfall,
                                  snow_load_study: Crocus):
    ratios = []
    for year in snow_load_study.ordered_years:
        maxima_index = snow_load_study.year_to_annual_maxima_index[year]
        snowfall = np.diagonal(snowfall_study.year_to_daily_time_serie_array[year][maxima_index])
        rainfall = np.diagonal(rainfall_study.year_to_daily_time_serie_array[year][maxima_index])
        ratio = compute_ratio(rainfall, snowfall)
        ratios.append(pd.Series(ratio))
    df = pd.concat(ratios, axis=1)
    df.index = rainfall_study.study_massif_names
    df.columns = rainfall_study.ordered_years
    return df


def df_snow_load_top_maxima_partition(nb_top_values, rainfall_study: SafranRainfall, snowfall_study: SafranSnowfall,
                                      snow_load_study: Crocus):
    ratios = []
    for i, (massif_name, ordered_index) in enumerate(
            snow_load_study.massif_name_to_annual_maxima_ordered_index.items()):
        ordered_years = snow_load_study.massif_name_to_annual_maxima_ordered_years[massif_name]
        top_ordered_index = ordered_index[-nb_top_values:]
        top_ordered_years = ordered_years[-nb_top_values:]
        # Top values only
        snowfall = [snowfall_study.year_to_daily_time_serie_array[year][idx, i] for idx, year in
                    zip(top_ordered_index, top_ordered_years)]
        rainfall = [rainfall_study.year_to_daily_time_serie_array[year][idx, i] for idx, year in
                    zip(top_ordered_index, top_ordered_years)]
        ratio = compute_ratio(np.array(rainfall), np.array(snowfall))
        ratios.append(pd.Series(ratio))
    df = pd.concat(ratios, axis=1).transpose()
    df.index = rainfall_study.study_massif_names
    return df


def main_snow_load_maxima_partition(year_min, year_max):
    rainfall_classes = [SafranRainfall1Day, SafranRainfall3Days, SafranRainfall5Days, SafranRainfall7Days]
    snowfall_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days]
    snow_load_classes = [CrocusSnowLoad1Day, CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days]
    classes = list(zip(rainfall_classes, snowfall_classes, snow_load_classes))[1:2]
    nb_top = 5
    for study_classes in classes:
        altitude_to_s = OrderedDict()
        for altitude in ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST[:]:
            studies = [study_class(altitude=altitude, year_min=year_min, year_max=year_max) for study_class in
                       study_classes]
            df = df_snow_load_top_maxima_partition(nb_top, *studies)
            df2 = df.transpose().describe().transpose()
            # print(df, '\n')
            # print(df2.iloc[:, 1:-2])
            df2.drop(columns=['count', 'std'], inplace=True)
            s = df2.mean(axis=0)
            # print(s)
            altitude_to_s[altitude] = s
        df_final = pd.DataFrame(altitude_to_s).transpose().round(2)
        print(nb_top, year_min, year_max)
        print(df_final)
        print('\n\n')


if __name__ == '__main__':
    main_snow_load_maxima_partition(1959, 2019)
    # main_snow_load_maxima_partition(1959, 1989)
    # main_snow_load_maxima_partition(1990, 2019)
