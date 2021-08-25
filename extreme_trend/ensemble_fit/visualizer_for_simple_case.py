from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import List, Dict

import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.adamont_data.cmip5.temperature_to_year import get_interval_limits, \
    get_year_min_and_year_max, get_ticks_labels_for_interval
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import \
    get_color_and_linestyle_from_massif_id
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh_with_multiprocessing_for_large_samples
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    GumbelTemporalModel
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitude_group import get_altitude_class_from_altitudes, \
    get_linestyle_for_altitude_class, get_altitude_group_from_altitudes
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.projected_extreme_snowfall.results.combination_utils import load_combination_name, \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.part_2.sensitivity_calibration_graph import short_name_to_label
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
                 weight_on_observation=1,
                 linear_effects=(False, False, False),
                 year_max_for_studies=None,
                 last_year_for_the_train_set=2019,
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
        self.last_year_for_the_train_set = last_year_for_the_train_set

        # Load the gcm rcm couple to studies
        if year_max_for_studies is None:
            gcm_to_year_min_and_year_max = None
        else:
            gcm_to_year_min_and_year_max = {gcm: (None, year_max_for_studies) for gcm in gcm_to_color.keys()}
        gcm_rcm_couple_to_studies = VisualizerForProjectionEnsemble.load_gcm_rcm_couple_to_studies(self.altitudes,
                                                                                                   self.gcm_rcm_couples,
                                                                                                   gcm_to_year_min_and_year_max,
                                                                                                   self.safran_study_class,
                                                                                                   self.scenario,
                                                                                                   self.season,
                                                                                                   self.study_class,
                                                                                                   year_max_for_safran_study=self.last_year_for_the_train_set)

        # Add the first 50% and the last 50% of the data
        self.other_obs_visualizers = []
        studies = AltitudesStudies(safran_study_class, altitudes, season=season, year_min=1959, year_max=2019)
        visu = AltitudesStudiesVisualizerForNonStationaryModels(studies,
                                                                model_classes=self.model_classes,
                                                                massif_names=[massif_name],
                                                                fit_method=fit_method,
                                                                temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                                display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
                                                                confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                                                                remove_physically_implausible_models=remove_physically_implausible_models,
                                                                param_name_to_climate_coordinates_with_effects=None,
                                                                linear_effects=(False, False, False),
                                                                weight_on_observation=weight_on_observation)
        self.other_obs_visualizers.append(visu)
        studies1 = AltitudesStudies(safran_study_class, altitudes, season=season, year_min=1959, year_max=1988)
        studies2 = AltitudesStudies(safran_study_class, altitudes, season=season, year_min=1989, year_max=2019)
        for studies in [studies1, studies2]:
            if issubclass(self.model_classes[0], GumbelTemporalModel):
                model_class_simplified = GumbelTemporalModel
            else:
                model_class_simplified = StationaryTemporalModel
            visu = AltitudesStudiesVisualizerForNonStationaryModels(studies,
                                                                    model_classes=[model_class_simplified],
                                                                    massif_names=[massif_name],
                                                                    fit_method=fit_method,
                                                                    temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                                    display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
                                                                    confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                                                                    remove_physically_implausible_models=remove_physically_implausible_models,
                                                                    param_name_to_climate_coordinates_with_effects=None,
                                                                    linear_effects=(False, False, False),
                                                                    weight_on_observation=weight_on_observation)
            self.other_obs_visualizers.append(visu)

        # Load the separate fit
        self.independent_ensemble_fit = IndependentEnsembleFit([self.massif_name], gcm_rcm_couple_to_studies,
                                                               model_classes,
                                                               fit_method, temporal_covariate_for_fit,
                                                               display_only_model_that_pass_gof_test,
                                                               confidence_interval_based_on_delta_method,
                                                               remove_physically_implausible_models,
                                                               None)

        # Load the together approach without the observation
        gcm_rcm_couple_to_studies_without_obs = {k: v for k, v in gcm_rcm_couple_to_studies.items() if k[0] != None}
        visualizer_ensemble_without_obs = VisualizerNonStationaryEnsemble(
            gcm_rcm_couple_to_studies=gcm_rcm_couple_to_studies_without_obs,
            massif_names=[self.massif_name],
            model_classes=model_classes,
            fit_method=fit_method, temporal_covariate_for_fit=temporal_covariate_for_fit,
            display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
            confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
            remove_physically_implausible_models=remove_physically_implausible_models,
            param_name_to_climate_coordinates_with_effects=None,
            linear_effects=linear_effects,
            weight_on_observation=weight_on_observation)

        # Load all the together fit approaches with observations
        self.combination_name_to_visualizer_ensemble = {'without obs': visualizer_ensemble_without_obs}
        if combinations_for_together is not None:
            for combination in combinations_for_together:
                param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                    combination)
                combination_name = load_combination_name(param_name_to_climate_coordinates_with_effects)
                visualizer_ensemble = VisualizerNonStationaryEnsemble(
                    gcm_rcm_couple_to_studies=gcm_rcm_couple_to_studies,
                    massif_names=[self.massif_name],
                    model_classes=model_classes,
                    fit_method=fit_method, temporal_covariate_for_fit=temporal_covariate_for_fit,
                    display_only_model_that_pass_anderson_test=display_only_model_that_pass_gof_test,
                    confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                    remove_physically_implausible_models=remove_physically_implausible_models,
                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                    linear_effects=linear_effects,
                    weight_on_observation=weight_on_observation)
                self.combination_name_to_visualizer_ensemble[combination_name] = visualizer_ensemble

    def compute_prediction_score(self):
        for combination_name, visualizer_together in self.combination_name_to_visualizer_ensemble.items():
            best_estimator = visualizer_together.massif_name_to_one_fold_fit[self.massif_name].best_estimator


            studies_train = AltitudesStudies(self.safran_study_class, self.altitudes, season=self.season,
                                       year_max=self.last_year_for_the_train_set)

            studies_test = AltitudesStudies(self.safran_study_class, self.altitudes, season=self.season,
                                       year_min=self.last_year_for_the_train_set + 1)

            for studies in [studies_train, studies_test][1:]:

                dataset_test = studies.spatio_temporal_dataset(self.massif_name)
                df_coordinates_temp_for_test = best_estimator.load_coordinates_temp(dataset_test.coordinates,
                                                                                    for_fit=False)
                maxima_values = dataset_test.observations.maxima_gev
                coordinate_values = df_coordinates_temp_for_test.values
                nllh = compute_nllh_with_multiprocessing_for_large_samples(coordinate_values, maxima_values,
                                                                           best_estimator.margin_function_from_fit,
                                                                           True, True, False)
                print(combination_name, nllh / len(coordinate_values))

    def visualize_gev_parameters(self):
        gev_params = GevParams.PARAM_NAMES + [True]
        for k, gev_param in enumerate(gev_params):
            print(self.get_str(gev_param), 'plot')
            self.visualize_gev_parameter(gev_param, k)

    def get_value(self, one_fold_fit, c, gev_param):
        gev_params = one_fold_fit.best_margin_function_from_fit.get_params(c)
        if gev_param in GevParams.PARAM_NAMES:
            return gev_params.to_dict()[gev_param]
        elif gev_param is True:
            return gev_params.mean
        else:
            raise NotImplementedError

    def get_str(self, gev_param):
        if gev_param in GevParams.PARAM_NAMES:
            return '{} parameter'.format(gev_param)
        elif gev_param is True:
            return "Mean"
        else:
            raise NotImplementedError

    def visualize_gev_parameter(self, gev_param, k):
        ax = plt.gca()
        # Independent plot
        items = list(self.independent_ensemble_fit.gcm_rcm_couple_to_visualizer.items())
        for vizu in self.other_obs_visualizers:
            items.append(((None, None), vizu))
        # items.append(((None, None), self.half_visualizers[0]))
        # items.append(((None, None), self.half_visualizers[1]))

        add_label_gcm = True
        for gcm_rcm_couple, visualizer in items:
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
            coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
            x = [c[0] for c in coordinates]
            y = [self.get_value(one_fold_fit, c, gev_param) for c in coordinates]
            if gcm_rcm_couple[0] is None:
                year_max = visualizer.study.ordered_years[-1]
                percentage = round(100 * (int(year_max) + 1 - 1959) / 61, 2)
                percentage = round(percentage / 10) * 10

                label = "non-stationary GEV for {}\% of the observation".format(percentage)
                linestyle = '-'
                linewidth = 3
                # percentage_to_marker = {
                #     70: 'x',
                #     100: None,
                # }
                # marker = percentage_to_marker[percentage]
            else:
                linestyle = '--'
                linewidth = 1
                if add_label_gcm:
                    add_label_gcm = False
                    label = "non-stationary GEV for one GCM-RCM couple"
                else:
                    label = None

            ax.plot(x, y, label=label, linestyle=linestyle, color='k', linewidth=linewidth, marker=None)

        # Together plot with obs
        colors = ['grey', 'blue', 'yellow', "orange", "red", "violet", 'gold']
        labels = ["Baseline", "Zero adjustment coefficients", 'One adjustment coefficient for all GCM-RCM pairs',
                  'One adjustment coefficient for each GCM', "One adjustment coefficient for each RCM",
                  'One adjustment coefficient for each GCM-RCM pair', 'one for all for the 3 parameters']
        for j, (combination_name, visualizer) in enumerate(self.combination_name_to_visualizer_ensemble.items()):
            color = colors[j]
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
            coordinates = one_fold_fit.best_estimator.coordinates_for_nllh
            x = sorted([c[0] for c in coordinates])
            y = [self.get_value(one_fold_fit, np.array([e]), gev_param) for e in x]
            # label = combination_name.replace('_', '-')
            label = labels[j]
            if (k < 3) and self.linear_effects[k]:
                if "no effect" not in label:
                    label += ' with linear effect'
            ax.plot(x, y, label=label, color=color, linewidth=3)
            # Add the slope with the added adjustment coefficients.
            # other_combinations = set([tuple(c[1:]) for c in coordinates])
            # other_combinations = [c for c in other_combinations if sum(c) > 0]
            # for last_coordinates in other_combinations:
            #     other_coordinates = [np.array([e] + list(last_coordinates)) for e in x]
            #     y = [one_fold_fit.best_margin_function_from_fit.get_params(c).to_dict()[gev_param] for c in
            #          other_coordinates]
            #     ax.plot(x, y, linestyle='--', color=color)

        # Final plt
        ylabel = '{} ({})'.format(self.get_str(gev_param), visualizer.study.variable_unit)
        ylabel = ylabel[0].upper() + ylabel[1:]
        ax.set_ylabel(ylabel)
        xlabel = 'T, the smoothed anomaly of global temperature w.r.t. pre-industrial levels (K)'
        ax.set_xlabel(xlabel)
        ax.legend()

        title = '{} massif {}'.format(self.massif_name, self.get_str(gev_param))
        visualizer.plot_name = title
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()
