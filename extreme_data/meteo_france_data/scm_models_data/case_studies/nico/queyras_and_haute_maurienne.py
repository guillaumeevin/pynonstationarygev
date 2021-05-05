from collections import OrderedDict

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusWetth
from extreme_data.meteo_france_data.scm_models_data.utils import ORIENTATIONS, orientation_float_to_orientation_name
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import ALL_ALTITUDES_WITHOUT_NAN


def create_csv():
    """
    before running
    we need to change in abstract study to do:
        REANALYSIS_ALPS_FLAT_FOLDER = 'SAFRAN_montagne-CROCUS_2019/alp_flat/reanalysis'

    :return:
    """
    for massif_name in ['Queyras', 'Haute-Maurienne'][:]:
        print(massif_name)
        writer = pd.ExcelWriter('{}.xlsx'.format(massif_name), engine='xlsxwriter')
        for altitude in ALL_ALTITUDES_WITHOUT_NAN[:]:
            write_one_sheet_name(writer, massif_name, altitude)
        writer.save()


def write_one_sheet_name(writer, massif_name, altitude):
    print(altitude)
    column_name_to_values = OrderedDict()
    slope = 40.0
    l = [(CrocusDepth, 'Depth'), (CrocusWetth, 'Ramsond'), (CrocusWetth, 'WetTh')]
    # Flat values
    study = add_to_dict(altitude, column_name_to_values, l, massif_name, None, "Flat", slope)
    # Slope values
    for orientation in ORIENTATIONS[:]:
        orientation_name = orientation_float_to_orientation_name[orientation]
        study = add_to_dict(altitude, column_name_to_values, l, massif_name, orientation, orientation_name, slope)
    if len(column_name_to_values) > 0:
        df = pd.DataFrame(column_name_to_values, index=study.all_days)
        df.to_excel(writer, sheet_name='{} m'.format(altitude))


def add_to_dict(altitude, column_name_to_values, l, massif_name, orientation, orientation_name, slope):
    for study_class, name in l:
        study = study_class(altitude=altitude, orientation=orientation, slope=slope)
        column_name = "{} {} {} degrees".format(name, orientation_name, slope)
        if massif_name in study.study_massif_names:
            daily_time_series = study.massif_name_to_daily_time_series[massif_name]
            # print(len(daily_time_series) / 365)
            column_name_to_values[column_name] = daily_time_series
    return study


if __name__ == '__main__':
    create_csv()
