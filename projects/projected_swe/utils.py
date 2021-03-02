import os

from collections import OrderedDict

import pandas as pd
import os.path as op
import datetime
import time
import numpy as np
from scipy.special import softmax

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples, \
    gcm_rcm_couple_to_str, SEPARATOR_STR, scenario_to_str, str_to_gcm_rcm_couple
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.utils import DATA_PATH
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import compute_nllh
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate

WEIGHT_COLUMN_NAME = "all weights"

WEIGHT_FOLDER = "ensemble weight"


def get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario):
    nb_gcm_rcm_couples = len(gcm_rcm_couples)
    nb_altitudes_list = len(altitudes_list)
    ensemble_folder_path = op.join(DATA_PATH, WEIGHT_FOLDER)
    if not op.exists(ensemble_folder_path):
        os.makedirs(ensemble_folder_path, exist_ok=True)
    scenario_str = scenario_to_str(scenario)
    csv_filename = "weights_{}_{}_{}_{}_{}.csv" \
        .format(nb_gcm_rcm_couples, nb_altitudes_list, year_min, year_max, scenario_str)
    weight_csv_filepath = op.join(ensemble_folder_path, csv_filename)
    return weight_csv_filepath


def save_to_filepath(df, gcm_rcm_couples, altitudes_list,
                     year_min, year_max,
                     scenario):
    filepath = get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario)
    df = df.round(decimals=3)
    df.index = [gcm_rcm_couple_to_str(i) for i in df.index]
    df.columns = [gcm_rcm_couple_to_str(i) if j > 0 else i for j, i in enumerate(df.columns)]

    # df.columns = [gcm_rcm_couple_to_str(i) for i in df.index]
    print(df.head())
    df.to_csv(filepath)


def load_gcm_rcm_couple_to_weight(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario,
                                  gcm_rcm_couple_missing=None):
    filepath = get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max, scenario)
    df = pd.read_csv(filepath, index_col=0)
    df.index = [str_to_gcm_rcm_couple(i) for i in df.index]
    df.columns = [str_to_gcm_rcm_couple(i) if j > 0 else i for j, i in enumerate(df.columns)]
    if gcm_rcm_couple_missing is None:
        column_name = WEIGHT_COLUMN_NAME
    else:
        column_name = gcm_rcm_couple_missing
    d = df[column_name].to_dict()
    return d
