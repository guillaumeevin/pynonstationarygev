import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.altitude_group import get_altitude_group_from_altitudes
from extreme_trend.one_fold_fit.plots.plot_histogram_altitude_studies import \
    plot_histogram_all_trends_against_altitudes, plot_shoe_plot_changes_against_altitude
from projects.projected_extreme_snowfall.results.part_2.average_bias import load_study
from projects.projected_extreme_snowfall.results.part_3.bias_reduction import plot_bias_reduction
from projects.projected_extreme_snowfall.results.part_3.plot_gcm_rcm_effects import plot_gcm_rcm_effects, \
    load_total_effect
from projects.projected_extreme_snowfall.results.part_3.plot_relative_change_in_return_level import \
    plot_relative_dynamic
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class VisualizerForProjectionEnsemble(object):

    def __init__(self, altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel],
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 gcm_to_year_min_and_year_max=None,
                 interval_str_prefix='',
                 safran_study_class=None,
                 param_name_to_climate_coordinates_with_effects=None,
                 linear_effects=(False, False, False),
                 ):
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.study_class = study_class
        self.safran_study_class = safran_study_class
        self.interval_str_prefix = interval_str_prefix
        self.altitudes_list = altitudes_list
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.scenario = scenario
        self.gcm_rcm_couples = gcm_rcm_couples
        self.massif_names = massif_names
        self.ensemble_fit_classes = ensemble_fit_classes

        # Some checks
        if gcm_to_year_min_and_year_max is not None:
            for gcm, years in gcm_to_year_min_and_year_max.items():
                assert isinstance(gcm, str), gcm
                assert len(years) == 2, years

        # Load all studies
        altitude_group_to_gcm_couple_to_studies = OrderedDict()
        for altitudes in altitudes_list:
            altitude_group = get_altitude_group_from_altitudes(altitudes)
            gcm_rcm_couple_to_studies = self.load_gcm_rcm_couple_to_studies(altitudes, gcm_rcm_couples,
                                                                            gcm_to_year_min_and_year_max,
                                                                            safran_study_class, scenario, season,
                                                                            study_class)
            altitude_group_to_gcm_couple_to_studies[altitude_group] = gcm_rcm_couple_to_studies

        # Load ensemble fit
        self.altitude_group_to_ensemble_class_to_ensemble_fit = OrderedDict()
        for altitude_group, gcm_rcm_couple_to_studies in altitude_group_to_gcm_couple_to_studies.items():
            ensemble_class_to_ensemble_fit = {}
            for ensemble_fit_class in ensemble_fit_classes:
                ensemble_fit = ensemble_fit_class(massif_names, gcm_rcm_couple_to_studies, model_classes,
                                                  fit_method, temporal_covariate_for_fit,
                                                  display_only_model_that_pass_gof_test,
                                                  confidence_interval_based_on_delta_method,
                                                  remove_physically_implausible_models,
                                                  param_name_to_climate_coordinates_with_effects,
                                                  linear_effects)
                ensemble_class_to_ensemble_fit[ensemble_fit_class] = ensemble_fit
            self.altitude_group_to_ensemble_class_to_ensemble_fit[altitude_group] = ensemble_class_to_ensemble_fit

    @classmethod
    def load_gcm_rcm_couple_to_studies(cls, altitudes, gcm_rcm_couples, gcm_to_year_min_and_year_max,
                                       safran_study_class, scenario, season, study_class,
                                       year_max_for_safran_study=None):
        gcm_rcm_couple_to_studies = {}
        for gcm_rcm_couple in gcm_rcm_couples:
            if gcm_to_year_min_and_year_max is None:
                kwargs_study = {}
            else:
                gcm = gcm_rcm_couple[0]
                if gcm not in gcm_to_year_min_and_year_max:
                    # It means that for this gcm and scenario,
                    # there is not enough data (less than 30 years) for the fit
                    continue
                year_min, year_max = gcm_to_year_min_and_year_max[gcm]
                kwargs_study = {'year_min': year_min, 'year_max': year_max}
            studies = AltitudesStudies(study_class, altitudes, season=season,
                                       scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                                       **kwargs_study)
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = studies
        # Potentially add the observations
        if safran_study_class is not None:
            if year_max_for_safran_study is not None:
                studies = AltitudesStudies(safran_study_class, altitudes, season=season,
                                           year_max=year_max_for_safran_study)
            else:
                studies = AltitudesStudies(safran_study_class, altitudes, season=season)
            gcm_rcm_couple_to_studies[(None, None)] = studies
        if len(gcm_rcm_couple_to_studies) == 0:
            print('No valid studies for the following couples:', gcm_rcm_couples)
        return gcm_rcm_couple_to_studies

    @property
    def has_elevation_non_stationarity(self):
        return all([len(a) > 1 for a in self.altitudes_list])

    def plot_for_visualizer_list(self, visualizer_list):
        if self.has_elevation_non_stationarity:
            with_significance = False
            for visualizer in visualizer_list:
                visualizer.plot_moments(with_significance=with_significance)
            plot_histogram_all_trends_against_altitudes(self.massif_names, visualizer_list,
                                                        with_significance=with_significance)
            for relative in [True, False]:
                plot_shoe_plot_changes_against_altitude(self.massif_names, visualizer_list, relative=relative,
                                                        with_significance=with_significance)
        else:
            with_significance = False
            # Correction coefficient plots
            # if self.param_name_to_climate_coordinates_with_effects is not None:
            #     # Plot the bias in the mean and std after taking into account the bias correction
            #     for visualizer in visualizer_list:
            #         for massif_name in self.massif_names:
            #             gcm_rcm_couple_to_study, safran_study = load_study(visualizer.studies.study.altitude,
            #                                                                self.gcm_rcm_couples,
            #                                                                self.safran_study_class, self.scenario,
            #                                                                self.study_class)
            #
            #             gcm_rcm_couple_to_params_effects = {}
            #             for gcm_rcm_couple in self.gcm_rcm_couples:
            #                 params_effects = [load_total_effect(gcm_rcm_couple, massif_name,
            #                                                     param_name, visualizer)
            #                                   for param_name in GevParams.PARAM_NAMES]
            #                 gcm_rcm_couple_to_params_effects[gcm_rcm_couple] = params_effects
            #             plot_bias_reduction(gcm_rcm_couple_to_study, massif_name, safran_study, visualizer, self.scenario)
            #     if len(visualizer_list) > 1:
            #         self.plot_effect_against_altitude(visualizer_list)
            # Moment plot
            for relative in [None, True, False][:1]:
                orders = [None] + GevParams.PARAM_NAMES[:]
                for order in orders[:1]:
                    plot_relative_dynamic(self.massif_names, visualizer_list,
                                          self.param_name_to_climate_coordinates_with_effects,
                                          self.safran_study_class,
                                          relative,
                                          order,
                                          self.gcm_rcm_couples,
                                          with_significance)

    def plot_effect_against_altitude(self, visualizer_list):
        climate_coordinate_with_effects_to_list = {
            (AbstractCoordinates.COORDINATE_GCM, AbstractCoordinates.COORDINATE_RCM): self.gcm_rcm_couples,
            AbstractCoordinates.COORDINATE_GCM: [[e] for e in set([g for g, r in self.gcm_rcm_couples])],
            AbstractCoordinates.COORDINATE_RCM: [[e] for e in set([r for g, r in self.gcm_rcm_couples])]
        }
        for c, gcm_rcm_couples in climate_coordinate_with_effects_to_list.items():
            for param_name in GevParams.PARAM_NAMES[:]:
                climate_coordinates_with_param_effects = self.param_name_to_climate_coordinates_with_effects[param_name]
                if climate_coordinates_with_param_effects is not None:
                    climate_coordinates_names_with_param_effects_to_extract = list(c) if isinstance(c, tuple) else [c]
                    if set(climate_coordinates_names_with_param_effects_to_extract).issubset(
                            set(climate_coordinates_with_param_effects)):
                        plot_gcm_rcm_effects(self.massif_names, visualizer_list,
                                             climate_coordinates_names_with_param_effects_to_extract,
                                             self.safran_study_class,
                                             gcm_rcm_couples,
                                             param_name)

    def plot(self):
        # Set limit for the plot
        visualizer_list = []
        for ensemble_fit_class in self.ensemble_fit_classes:
            for ensemble_fit in self.ensemble_fits(ensemble_fit_class):
                visualizer_list.extend(ensemble_fit.visualizer_list)
        # compute_and_assign_max_abs(visualizer_list)
        # Plot
        if IndependentEnsembleFit in self.ensemble_fit_classes:
            self.plot_independent()
        if TogetherEnsembleFit in self.ensemble_fit_classes:
            self.plot_together()

    def plot_independent(self):
        # Aggregated at gcm_rcm_level plots
        merge_keys = [AbstractEnsembleFit.Median_merge, AbstractEnsembleFit.Mean_merge]
        keys = self.gcm_rcm_couples + merge_keys
        # Only plot Mean for speed
        # keys = [AbstractEnsembleFit.Mean_merge]
        for key in keys:
            visualizer_list = [independent_ensemble_fit.gcm_rcm_couple_to_visualizer[key]
                               if key in self.gcm_rcm_couples
                               else independent_ensemble_fit.merge_function_name_to_visualizer[key]
                               for independent_ensemble_fit in self.ensemble_fits(IndependentEnsembleFit)
                               ]
            if key in merge_keys:
                for v in visualizer_list:
                    v.studies.study.gcm_rcm_couple = ("{} {}".format(key, "merge"), self.interval_str_prefix)
            self.plot_for_visualizer_list(visualizer_list)

    def plot_together(self):
        visualizer_list = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in self.ensemble_fits(TogetherEnsembleFit)]
        for v in visualizer_list:
            v.studies.study.gcm_rcm_couple = ("together merge", self.interval_str_prefix)
        self.plot_for_visualizer_list(visualizer_list)

    def ensemble_fits(self, ensemble_class):
        """Return the ordered ensemble fit for a given ensemble class (in the order of the altitudes)"""
        return [ensemble_class_to_ensemble_fit[ensemble_class]
                for ensemble_class_to_ensemble_fit
                in self.altitude_group_to_ensemble_class_to_ensemble_fit.values()]

    def plot_preliminary_first_part(self):
        if self.massif_names is None:
            massif_names = AbstractStudy.all_massif_names()
        else:
            massif_names = self.massif_names
        assert isinstance(massif_names, list)
        # Plot for all parameters
        for param_name in GevParams.PARAM_NAMES:
            for degree in [0, 1]:
                for massif_name in massif_names:
                    self.plot_preliminary_first_part_for_one_massif(massif_name, param_name, degree)

    def plot_preliminary_first_part_for_one_massif(self, massif_name, param_name, degree):
        # Retrieve the data
        ensemble_fit: IndependentEnsembleFit
        gcm_rcm_couple_to_data = {
            c: [] for c in self.gcm_rcm_couples
        }
        for ensemble_fit in self.ensemble_fits(IndependentEnsembleFit):
            for gcm_rcm_couple in self.gcm_rcm_couples:
                visualizer = ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple]
                if massif_name in visualizer.massif_name_to_one_fold_fit:
                    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
                    coef = one_fold_fit.best_coef(param_name, 0, degree)
                    altitude = visualizer.altitude_group.reference_altitude
                    gcm_rcm_couple_to_data[gcm_rcm_couple].append((altitude, coef))
        # Plot
        ax = plt.gca()
        for gcm_rcm_couple, data in gcm_rcm_couple_to_data.items():
            altitudes, coefs = list(zip(*data))
            color = gcm_rcm_couple_to_color[gcm_rcm_couple]
            label = gcm_rcm_couple_to_str(gcm_rcm_couple)
            ax.plot(coefs, altitudes, color=color, label=label, marker='o')
        ax.legend()
        visualizer.plot_name = '{}/{}_{}'.format(param_name, degree, massif_name)
        visualizer.show_or_save_to_file(no_title=True)
        plt.close()
