from collections import OrderedDict

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepthIn3Days, CrocusDepthWet, CrocusDepth
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature, SafranSnowfall1Day, \
    SafranPrecipitation1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import ALL_ALTITUDES

massifs_names = ['Queyras', 'Thabor', 'Haute-Maurienne'][:]
study_classes = [SafranTemperature, SafranSnowfall1Day, SafranPrecipitation1Day,
                 CrocusDepth, CrocusDepthIn3Days, CrocusDepthWet][:]
altitudes = ALL_ALTITUDES
season = Season.annual
for massif_name in massifs_names:
    writer = pd.ExcelWriter('{}.xlsx'.format(massif_name), engine='xlsxwriter')
    for study_class in study_classes:
        altitude_to_time_serie = OrderedDict()
        for altitude in altitudes:
            study = study_class(altitude=altitude)  # type: AbstractStudy
            if massif_name in study.study_massif_names:
                massif_id = study.massif_name_to_massif_id[massif_name]
                time_serie = study.all_daily_series[:, massif_id]
                altitude_to_time_serie[altitude] = time_serie
        df = pd.DataFrame(altitude_to_time_serie, columns=altitudes, index=study.all_days)
        print(df.head())
        df.to_excel(writer, sheet_name=study.variable_name)
    writer.save()



