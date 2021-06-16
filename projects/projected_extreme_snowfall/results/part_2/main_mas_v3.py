from collections import OrderedDict
import os.path as op
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects, load_combination_name_for_tuple
from projects.projected_extreme_snowfall.results.experiment.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_2.plot_bias import plot_bias, plot_average_bias, load_study
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.part_2.v2.utils import update_csv, is_already_done
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load


def main_preliminary_projections():
    # Load parameters
    fast = False
    snowfall = False
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class = set_up_and_load(
        fast, snowfall)
    display_only_model_that_pass_gof_test = False
    # Load study
    massif_name = massif_names[0]
    altitudes = altitudes_list[0]
    altitude = altitudes[0]
    gcm_rcm_couple_to_study, safran_study = load_study(altitude, gcm_rcm_couples, safran_study_class, scenario,
                                                       study_class)
    print('number of couples loaded:', len(gcm_rcm_couple_to_study))
    average_bias = plot_bias(gcm_rcm_couple_to_study, massif_name, safran_study)

    if fast in [True, None]:
        alpha = ''
        gcm_rcm_couples_sampled_for_experiment = [('CNRM-CM5', 'ALADIN63')]
        gcm_rcm_couples_sampled_for_experiment = [('NorESM1-M', 'REMO2015'), ('MPI-ESM-LR', 'REMO2009')]
    else:
        alpha = 40
        gcm_rcm_couples_sampled_for_experiment = plot_average_bias(gcm_rcm_couple_to_study, massif_name, average_bias,
                                                                   alpha)

    print(gcm_rcm_couples_sampled_for_experiment)

    csv_filename = 'last_snow_load_fast_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_alpha_{}.xlsx'.format(fast, altitude,
                                                                                                     len(model_classes),
                                                                                                     len(gcm_rcm_couple_to_study),
                                                                                                     alpha)
    print(csv_filename)
    csv_filepath = op.join(CSV_PATH, csv_filename)

    print("Number of couples:", len(gcm_rcm_couples_sampled_for_experiment))

    idx_list = [0, 1, 2, 3, 4, 5][1:]
    for i in idx_list:
        j = i
        print(i, j)
        for gcm_rcm_couple in gcm_rcm_couples_sampled_for_experiment:
            combination = (i, j, 0)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination)

            combination_name = load_combination_name_for_tuple(combination)
            if is_already_done(csv_filepath, combination_name, altitude, gcm_rcm_couple):
                continue
            xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                        scenario=scenario,
                                        selection_method_names=['aic'],
                                        model_classes=model_classes,
                                        massif_names=massif_names,
                                        fit_method=MarginFitMethod.evgam,
                                        temporal_covariate_for_fit=temporal_covariate_for_fit,
                                        remove_physically_implausible_models=remove_physically_implausible_models,
                                        display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                        gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment,
                                        param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                        )
            nllh_list = xp.run_one_experiment(gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)
            update_csv(csv_filepath, combination_name, altitude, gcm_rcm_couple, nllh_list)


if __name__ == '__main__':
    main_preliminary_projections()
