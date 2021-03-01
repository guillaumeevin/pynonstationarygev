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
    gcm_rcm_couple_to_str, SEPARATOR_STR
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.utils import DATA_PATH
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import compute_nllh
from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate

WEIGHT_COLUMN_NAME = "weight"

WEIGHT_FOLDER = "ensemble weight"


def get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max):
    nb_gcm_rcm_couples = len(gcm_rcm_couples)
    nb_altitudes_list = len(altitudes_list)
    ensemble_folder_path = op.join(DATA_PATH, WEIGHT_FOLDER)
    if not op.exists(ensemble_folder_path):
        os.makedirs(ensemble_folder_path, exist_ok=True)
    csv_filename = "weights_{}_{}_{}_{}.csv".format(nb_gcm_rcm_couples, nb_altitudes_list, year_min, year_max)
    weight_csv_filepath = op.join(ensemble_folder_path, csv_filename)
    return weight_csv_filepath


def load_gcm_rcm_couple_to_weight(gcm_rcm_couples, altitudes_list, year_min, year_max):
    filepath = get_csv_filepath(gcm_rcm_couples, altitudes_list, year_min, year_max)
    df = pd.read_csv(filepath, index_col=0)
    d = df[WEIGHT_COLUMN_NAME].to_dict()
    d = {tuple(k.split(SEPARATOR_STR)): v for k, v in d.items()}
    return d


def save_gcm_rcm_couple_to_weight(visualizer: VisualizerForProjectionEnsemble, scm_study_class,
                                  year_min, year_max):
    gcm_rcm_couple_to_nllh_sum = OrderedDict()
    for c in visualizer.gcm_rcm_couples:
        gcm_rcm_couple_to_nllh_sum[c] = 0
    for ensemble_fit in visualizer.ensemble_fits(ensemble_class=IndependentEnsembleFit):
        # Load the AltitudeStudies
        scm_studies = AltitudesStudies(scm_study_class, ensemble_fit.altitudes, year_min=year_min,year_max=year_max)
        for altitude, study in scm_studies.altitude_to_study.items():
            for massif_name, maxima in study.massif_name_to_annual_maxima.items():
                # Check that all the gcm_rcm_couple have a model for this massif_name
                if condition_to_compute_nllh(ensemble_fit, massif_name, visualizer):
                    print(ensemble_fit.altitudes, massif_name)
                    coordinates = [np.array([altitude, year]) for year in study.ordered_years]
                    nllh_list = []
                    for gcm_rcm_couple in visualizer.gcm_rcm_couples:
                        best_function_from_fit = get_function_from_fit(ensemble_fit, massif_name, gcm_rcm_couple)
                        # It is normal that it could crash, because some models where fitted with data smaller than
                        # the data used to compute the nllh
                        nllh = compute_nllh(coordinates, maxima, best_function_from_fit,
                                            maximum_from_obs=False, assertion_for_inf=False)
                        nllh_list.append(nllh)
                    if all([not np.isinf(nllh) for nllh in nllh_list]):
                        for nllh, gcm_rcm_couple in zip(nllh_list, visualizer.gcm_rcm_couples):
                            gcm_rcm_couple_to_nllh_sum[gcm_rcm_couple] += nllh

    # Compute the final weight
    print(gcm_rcm_couple_to_nllh_sum)
    llh_list = -np.array(list(gcm_rcm_couple_to_nllh_sum.values()))
    weights = softmax(llh_list)
    couple_names = [gcm_rcm_couple_to_str(c) for c in visualizer.gcm_rcm_couples]
    gcm_rcm_couple_to_normalized_weights = dict(zip(couple_names, weights))
    print(gcm_rcm_couple_to_normalized_weights)
    # Save to csv
    filepath = get_csv_filepath(visualizer.gcm_rcm_couples, visualizer.altitudes_list, year_min, year_max)
    df = pd.DataFrame({WEIGHT_COLUMN_NAME: weights}, index=couple_names)
    print(df)
    df.to_csv(filepath)


def condition_to_compute_nllh(ensemble_fit, massif_name, visualizer):
    return all(
        [massif_name in ensemble_fit.gcm_rcm_couple_to_visualizer[c].massif_name_to_one_fold_fit for c in
         visualizer.gcm_rcm_couples])


def get_function_from_fit(ensemble_fit, massif_name, gcm_rcm_couple):
    visualizer = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple]
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    return one_fold_fit.best_function_from_fit


def main_weight_computation():
    start = time.time()
    study_class = AdamontSnowfall
    scm_study_class = {
        AdamontSnowfall: SafranSnowfall1Day,
    }[study_class]
    ensemble_fit_classes = [IndependentEnsembleFit]
    temporal_covariate_for_fit = TimeTemporalCovariate
    model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    remove_physically_implausible_models = True
    scenario = AdamontScenario.histo
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)
    year_min = 1982
    year_max = 2005
    # todo: maybe i should also limit the years for the fit of the model for each ensemble ?

    fast = None
    if fast is None:
        massif_names = None
        altitudes_list = altitudes_for_groups[2:3]
        gcm_rcm_couples = gcm_rcm_couples[:10]
    elif fast:
        massif_names = ['Pelvoux'][:1]
        altitudes_list = altitudes_for_groups[2:3]
        gcm_rcm_couples = gcm_rcm_couples[:2]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups[:]

    visualizer = VisualizerForProjectionEnsemble(
        altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
        model_classes=model_classes,
        ensemble_fit_classes=ensemble_fit_classes,
        massif_names=massif_names,
        temporal_covariate_for_fit=temporal_covariate_for_fit,
        remove_physically_implausible_models=remove_physically_implausible_models,
        gcm_to_year_min_and_year_max=None,
    )
    save_gcm_rcm_couple_to_weight(visualizer, scm_study_class, year_min, year_max)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_weight_computation()
    # d = load_gcm_rcm_couple_to_weight(['sd', 'sdf'], [23])
    # print(d)
