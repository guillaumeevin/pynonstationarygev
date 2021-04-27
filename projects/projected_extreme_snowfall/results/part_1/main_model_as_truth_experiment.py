import datetime
import time
from itertools import product

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.part_1.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


def main_model_as_truth_experiment():
    start = time.time()
    fast = True

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit = set_up_and_load(fast)

    climate_coordinates_with_effects_list = [[AbstractCoordinates.COORDINATE_GCM, AbstractCoordinates.COORDINATE_RCM],
                                             [AbstractCoordinates.COORDINATE_GCM],
                                             [AbstractCoordinates.COORDINATE_RCM],
                                             None
                                             ]  # None means we do not create any effect
    potential_indices = list(range(4))
    if fast is True:
        potential_indices = potential_indices[:1]

    selection_method_names = ['aic', 'aicc', 'bic']
    combination_to_average_predicted_nllh = {}
    for combination in product(*[potential_indices for _ in range(3)]):
        param_name_to_climate_coordinates_with_effects = {param_name: climate_coordinates_with_effects_list[idx]
                                                          for param_name, idx in
                                                          zip(GevParams.PARAM_NAMES, combination)}

    for climate_coordinates_with_effects in climate_coordinates_with_effects_list[:1]:
        print(climate_coordinates_with_effects)
        average = compute_average_nllh(altitudes_list, climate_coordinates_with_effects, gcm_rcm_couples, massif_names,
                                       model_classes, scenario, study_class, temporal_covariate_for_fit,
                                       selection_method_names)
        average_list.append(average)
    combination_to_average_predicted_nllh[selection_method_name] = average_list
    df = pd.DataFrame.from_dict(combination_to_average_predicted_nllh)
    print(df)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def compute_average_nllh(altitudes_list, climate_coordinates_with_effects, gcm_rcm_couples, massif_names, model_classes,
                         scenario, study_class, temporal_covariate_for_fit, selection_method_names):
    nllh_list = []
    for altitudes in altitudes_list[:1]:
        xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                    scenario=scenario,
                                    selection_method_names=selection_method_names,
                                    model_classes=model_classes,
                                    massif_names=massif_names,
                                    fit_method=MarginFitMethod.evgam,
                                    temporal_covariate_for_fit=temporal_covariate_for_fit,
                                    remove_physically_implausible_models=True,
                                    display_only_model_that_pass_gof_test=True,
                                    climate_coordinates_with_effects=climate_coordinates_with_effects,
                                    )
        nllh_list.append(xp.run_all_experiments())
    return np.mean(nllh_list)


if __name__ == '__main__':
    main_model_as_truth_experiment()
