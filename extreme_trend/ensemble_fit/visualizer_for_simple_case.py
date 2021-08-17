from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import List, Dict

import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.adamont_data.cmip5.temperature_to_year import get_interval_limits, \
    get_year_min_and_year_max, get_ticks_labels_for_interval
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import \
    get_color_and_linestyle_from_massif_id
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitude_group import get_altitude_class_from_altitudes, \
    get_linestyle_for_altitude_class, get_altitude_group_from_altitudes
from projects.projected_extreme_snowfall.results.combination_utils import load_combination_name, \
    load_param_name_to_climate_coordinates_with_effects
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


class VisualizerForSimpleCase(object):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 massif_name=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 combinations_for_together=None,
                 linear_effects=False
                 ):
        self.linear_effects = linear_effects
        self.combinations_for_together = combinations_for_together
        self.safran_study_class = safran_study_class
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        self.display_only_model_that_pass_gof_test = display_only_model_that_pass_gof_test
        self.model_classes = model_classes
        self.scenario = scenario
        self.season = season
        self.study_class = study_class
        self.gcm_rcm_couples = gcm_rcm_couples
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.altitudes = altitudes
        self.massif_name = massif_name

        # Load the gcm rcm couple to studies
        gcm_rcm_couple_to_studies = VisualizerForProjectionEnsemble.load_gcm_rcm_couple_to_studies(self.altitudes,
                                                                                                   self.gcm_rcm_couples,
                                                                                                   None,
                                                                                                   self.safran_study_class,
                                                                                                   self.scenario,
                                                                                                   self.season,
                                                                                                   self.study_class)
        # Load the separate fit
        self.independent_ensemble_fit = IndependentEnsembleFit([self.massif_name], gcm_rcm_couple_to_studies,
                                                               model_classes,
                                                               fit_method, temporal_covariate_for_fit,
                                                               display_only_model_that_pass_gof_test,
                                                               confidence_interval_based_on_delta_method,
                                                               remove_physically_implausible_models,
                                                               None)

        # Load all the together fit approaches
        self.combination_name_to_together_ensemble_fit = {}
        if combinations_for_together is not None:
            for combination in combinations_for_together:
                param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(combination)
                combination_name = load_combination_name(param_name_to_climate_coordinates_with_effects)
                together_ensemble_fit = TogetherEnsembleFit([self.massif_name], gcm_rcm_couple_to_studies,
                                                               model_classes,
                                                               fit_method, temporal_covariate_for_fit,
                                                               display_only_model_that_pass_gof_test,
                                                               confidence_interval_based_on_delta_method,
                                                               remove_physically_implausible_models,
                                                               param_name_to_climate_coordinates_with_effects,
                                                               linear_effects)
                self.combination_name_to_together_ensemble_fit[combination_name] = together_ensemble_fit

    def visualize_gev_parameters(self):
        for gev_param in GevParams.PARAM_NAMES[:]:
            print(gev_param, 'plot')
            self.visualize_gev_parameter(gev_param)

    def visualize_gev_parameter(self, gev_param):
        ax = plt.gca()
        # Independent plot
        for gcm_rcm_couple, visualizer in self.independent_ensemble_fit.gcm_rcm_couple_to_visualizer.items():
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
            coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
            x = [c[0] for c in coordinates]
            y = [one_fold_fit.best_margin_function_from_fit.get_params(c).to_dict()[gev_param] for c in coordinates]
            if gcm_rcm_couple[0] is None:
                label = "observation"
                linestyle = '-'
            else:
                linestyle = '--'
                label = None

            ax.plot(x, y, label=label, linestyle=linestyle, color='k', linewidth=3)

        # Together plot
        colors = ['red', 'blue', 'green', 'orange']
        for j, (combination_name, together_ensemble_fit) in enumerate(self.combination_name_to_together_ensemble_fit.items()):
            color  = colors[j]
            one_fold_fit = together_ensemble_fit.visualizer.massif_name_to_one_fold_fit[self.massif_name]
            coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
            x = sorted([c[0] for c in coordinates])
            y = [one_fold_fit.best_margin_function_from_fit.get_params(np.array([e])).to_dict()[gev_param] for e in x]
            label = combination_name.replace('_','-')
            ax.plot(x, y, label=label, color=color)
            other_combinations = set([tuple(c[1:]) for c in coordinates])
            other_combinations = [c for c in other_combinations if sum(c) > 0]
            for last_coordinates in other_combinations:
                other_coordinates = [np.array([e] + list(last_coordinates)) for e in x]
                y = [one_fold_fit.best_margin_function_from_fit.get_params(c).to_dict()[gev_param] for c in other_coordinates]
                ax.plot(x, y, linestyle='--', color=color)

            # for c in coordinates:
            #     print(c)
            # x = [c[0] for c in coordinates]
            # y = [one_fold_fit.best_margin_function_from_fit.get_params(c).to_dict()[gev_param] for c in coordinates]
            # label = combination_name
            # ax.plot(x, y, label=label)

        # Final plt
        ylabel = '{} parameter ({})'.format(gev_param, visualizer.study.variable_unit)
        ylabel = ylabel[0].upper() + ylabel[1:]
        ax.set_ylabel(ylabel)
        xlabel = 'T, the smoothed anomaly of global temperature w.r.t. pre-industrial levels (K)'
        ax.set_xlabel(xlabel)
        ax.legend()

        title = '{} parameter for the {} massif'.format(gev_param, self.massif_name)
        visualizer.plot_name = title
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()
