import os
import os.path as op

import pandas as pd

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str, scenario_to_str, \
    str_to_gcm_rcm_couple
from extreme_data.utils import DATA_PATH
from root_utils import get_display_name_from_object_type

WEIGHT_COLUMN_NAME = "all weights"

WEIGHT_FOLDER = "ensemble weight"


def get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario, weight_computer_class):
    nb_gcm_rcm_couples = len(gcm_rcm_couples)
    nb_altitudes_list = len(altitudes_list)
    ensemble_folder_path = op.join(DATA_PATH, WEIGHT_FOLDER)
    if not op.exists(ensemble_folder_path):
        os.makedirs(ensemble_folder_path, exist_ok=True)
    scenario_str = scenario_to_str(scenario)
    class_name = get_display_name_from_object_type(weight_computer_class)
    csv_filename = "weights_{}_{}_{}_{}_{}_{}.csv" \
        .format(nb_gcm_rcm_couples, nb_altitudes_list, year_min, year_max, scenario_str, class_name)
    weight_csv_filepath = op.join(ensemble_folder_path, csv_filename)
    return weight_csv_filepath


def save_to_filepath(df, gcm_rcm_couples, altitudes_list,
                     year_min, year_max,
                     scenario, weight_computer_class):
    filepath = get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario, weight_computer_class)
    df = df.round(decimals=3)
    df.index = [gcm_rcm_couple_to_str(i) for i in df.index]
    df.columns = [gcm_rcm_couple_to_str(i) if j > 0 else i for j, i in enumerate(df.columns)]

    # df.columns = [gcm_rcm_couple_to_str(i) for i in df.index]
    print(df.head())
    df.to_csv(filepath)


def load_gcm_rcm_couple_to_weight(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario,
                                  weight_class, gcm_rcm_couple_missing):
    filepath = get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario, weight_class)
    df = pd.read_csv(filepath, index_col=0)
    df.index = [str_to_gcm_rcm_couple(i) for i in df.index]
    df.columns = [str_to_gcm_rcm_couple(i) if j > 0 else i for j, i in enumerate(df.columns)]
    if gcm_rcm_couple_missing is None:
        column_name = WEIGHT_COLUMN_NAME
    else:
        column_name = gcm_rcm_couple_missing
    d = df[column_name].to_dict()
    return d
