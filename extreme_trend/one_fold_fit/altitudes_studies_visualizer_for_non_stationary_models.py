from math import ceil, floor
from math import ceil, floor
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_averaged_global_mean_temp
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import remove_the_extreme_colors
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gumbel.gumbel_gof import get_pvalue_anderson_darling_test
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.one_fold_fit.altitude_group import \
    get_altitude_group_from_altitudes
from extreme_trend.one_fold_fit.one_fold_fit import \
    OneFoldFit
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_anderson_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 linear_effects=(False, False, False),
                 gcm_rcm_couple_as_pseudo_truth=None,
                 weight_on_observation=1,
                 ):
        super().__init__(studies.study, show=show, save_to_file=not show)
        self.linear_effects = linear_effects
        self.weight_on_observation = weight_on_observation
        self.gcm_rcm_couple_as_pseudo_truth = gcm_rcm_couple_as_pseudo_truth
        self.studies = studies
        self.model_classes = model_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_test = display_only_model_that_pass_anderson_test
        self.massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        self.massif_name_to_massif_id = {m: i for i, m in enumerate(self.massif_names)}
        self.altitude_group = get_altitude_group_from_altitudes(self.studies.altitudes)
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects

        self.massif_name_to_massif_altitudes = {}
        # Load one fold fit
        self.load_one_fold_fit()

        # Cache
        self._method_name_and_order_to_massif_name_to_value = {}
        self._method_name_and_order_to_max_abs = {}
        self._max_abs_for_shape = None

    def load_one_fold_fit(self):
        self._massif_name_to_one_fold_fit = dict()
        for massif_name in self.massif_names:
            o = self.fit_one_fold(massif_name)
            if o is not None:
                self._massif_name_to_one_fold_fit[massif_name] = o

    def fit_one_fold(self, massif_name):
        # Load valid massif altitudes
        massif_altitudes = self.get_massif_altitudes(massif_name)
        # Save the massif altitudes only for those who pass the condition
        self.massif_name_to_massif_altitudes[massif_name] = massif_altitudes
        # Load dataset
        try:
            dataset = self.get_dataset(massif_altitudes, massif_name, self.gcm_rcm_couple_as_pseudo_truth)
        except AssertionError as e:
            print('Exception for {}'.format(massif_name.replace("_", "")))
            print(e.__repr__())
            return None

        if isinstance(self.model_classes, dict):
            model_classes = [self.model_classes[massif_name]]
        else:
            model_classes = self.model_classes

        if isinstance(self.param_name_to_climate_coordinates_with_effects,
                      dict) and massif_name in self.param_name_to_climate_coordinates_with_effects:
            param_name_to_climate_coordinates_with_effects = self.param_name_to_climate_coordinates_with_effects[
                massif_name]
        else:
            param_name_to_climate_coordinates_with_effects = self.param_name_to_climate_coordinates_with_effects

        old_fold_fit = OneFoldFit(massif_name, dataset, model_classes,
                                  self.study.year_min,
                                  self.study.year_max,
                                  self.fit_method,
                                  self.temporal_covariate_for_fit,
                                  self.altitude_group,
                                  self.display_only_model_that_pass_test,
                                  self.confidence_interval_based_on_delta_method,
                                  self.remove_physically_implausible_models,
                                  param_name_to_climate_coordinates_with_effects,
                                  self.linear_effects)
        return old_fold_fit

    def get_dataset(self, massif_altitudes, massif_name, gcm_rcm_couple_as_pseudo_truth=None):
        if (len(massif_altitudes) == 1) and (gcm_rcm_couple_as_pseudo_truth is None):
            dataset = self.studies.spatio_temporal_dataset_memoize(massif_name, massif_altitudes[0])
        else:
            dataset = self.studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=massif_altitudes,
                                                       gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple_as_pseudo_truth)
        return dataset

    moment_names = ['moment', 'changes_of_moment', 'relative_changes_of_moment'][:]
    orders = [1, 2, None][2:]

    def get_massif_altitudes(self, massif_name):
        return self._get_massif_altitudes(massif_name, self.studies)

    def _get_massif_altitudes(self, massif_name, studies):
        valid_altitudes = [altitude for altitude, study in studies.altitude_to_study.items()
                           if massif_name in study.study_massif_names]
        massif_altitudes = []
        for altitude in valid_altitudes:
            study = studies.altitude_to_study[altitude]
            annual_maxima = study.massif_name_to_annual_maxima[massif_name]
            percentage_of_non_zeros = 100 * np.count_nonzero(annual_maxima) / len(annual_maxima)
            if percentage_of_non_zeros > 90:
                massif_altitudes.append(altitude)
            else:
                # if isinstance(self.altitude_group, DefaultAltitudeGroup):
                print('time series excluded due number of zeros > 10\%')
                print(massif_name, altitude, percentage_of_non_zeros)
        return massif_altitudes

    @property
    def massif_name_to_one_fold_fit(self) -> Dict[str, OneFoldFit]:
        return {massif_name: old_fold_fit for massif_name, old_fold_fit in self._massif_name_to_one_fold_fit.items()}

    def plot_moments_projections_snowfall(self):
        OneFoldFit.COVARIATE_BEFORE_TEMPERATURE = 1
        # Standard plot
        for order in [1, None][:]:
            for covariate in [2, 4][:]:
                for moment_name in ['changes_of_moment', 'relative_changes_of_moment'][:]:
                    max_abs_change = 40.01 if 'relative' in moment_name else 25.01
                    OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
                    self.plot_map_moment_projections(moment_name, order, max_abs_change, snowfall=True)

    def plot_moments_projections_snowfall_discussion(self, scenario):
        # Compute some number for the discussion
        print('frei 2018 comparison')
        self.set_covariates((1981, 2010), (2070, 2099), scenario)
        self.plot_map_moment_projections('relative_changes_of_moment', order=1, only_print=True)
        print('moreno 2011 comparison')
        self.set_covariates((1960, 1990), (2070, 2100), scenario)
        OneFoldFit.return_period = 25
        self.plot_map_moment_projections('relative_changes_of_moment', order=None, only_print=True)

    def set_covariates(self, covariate_before, covariate_after, scenario):
        OneFoldFit.COVARIATE_BEFORE_TEMPERATURE = tuple(
            year_to_averaged_global_mean_temp(scenario, covariate_before[0], covariate_before[1]).values())
        OneFoldFit.COVARIATE_AFTER_TEMPERATURE = tuple(
            year_to_averaged_global_mean_temp(scenario, covariate_after[0], covariate_after[1]).values())

    def plot_moments_projections(self, scenario):
        default_covariate = OneFoldFit.COVARIATE_AFTER_TEMPERATURE
        OneFoldFit.COVARIATE_AFTER_TEMPERATURE = 1
        max_abs = self.plot_map_moment_projections("moment", None)
        for covariate in [2, 3, 4]:
            print("covariate", covariate)
            OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
            self.plot_map_moment_projections("moment", None, max_abs)
        OneFoldFit.COVARIATE_AFTER_TEMPERATURE = default_covariate

        # Standard plot
        for order in [1, None]:
            for covariate in [2, 3, 4]:
                OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
                self.plot_map_moment_projections('relative_changes_of_moment', order)

        # Compute some number for the discussion
        covariate_before = (1986, 2005)
        if isinstance(covariate_before, tuple):
            covariate_before = tuple(year_to_averaged_global_mean_temp(scenario, covariate_before[0], covariate_before[1]).values())
        OneFoldFit.COVARIATE_BEFORE_TEMPERATURE = covariate_before

        for order in [1, None]:
            for covariate in [(2031, 2050), (2080, 2099)]:
                if isinstance(covariate, tuple):
                    covariate = tuple(year_to_averaged_global_mean_temp(scenario, covariate[0], covariate[1]).values())
                OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
                # self.plot_map_moment_projections('changes_of_moment', None, with_significance)
                self.plot_map_moment_projections('relative_changes_of_moment', order)

    def method_name_and_order_to_d(self, method_name, order):
        c = (method_name, order, OneFoldFit.COVARIATE_AFTER_TEMPERATURE)
        if c not in self._method_name_and_order_to_massif_name_to_value:
            # Compute values
            massif_name_to_value = {}
            for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
                value = \
                    one_fold_fit.__getattribute__(method_name)([self.altitude_group.reference_altitude], order=order)[0]
                massif_name_to_value[massif_name] = value
            # Remove values
            if any([np.isinf(v) for v in massif_name_to_value.values()]):
                print("shape to large > 0.5, thus removing std that are infinite")
            massif_name_to_value = {m: v for m, v in massif_name_to_value.items()
                                    if not np.isinf(v)}
            # todo: i could remove here potential undefined parameters
            # Store it
            self._method_name_and_order_to_massif_name_to_value[c] = massif_name_to_value
        return self._method_name_and_order_to_massif_name_to_value[c]



    def plot_map_moment_projections(self, method_name, order, max_abs_change=None, snowfall=False,
                                    only_print=False):
        massif_name_to_value = self.method_name_and_order_to_d(method_name, order)
        massif_name_to_text = {}

        # Plot settings
        moment = ' '.join(method_name.split('_'))
        d_temperature = {'C': '{C}'}
        str_for_last_year = ' at +${}^o\mathrm{C}$' \
            if self.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate else ' in {}'
        str_for_last_year = str_for_last_year.format(OneFoldFit.COVARIATE_AFTER_TEMPERATURE, **d_temperature)
        moment = moment.replace('moment', '{}{}'.format(OneFoldFit.get_moment_str(order=order), str_for_last_year))
        plot_name = '{} '.format(moment)

        if 'change' in method_name:
            plot_name = plot_name.replace(str_for_last_year, '')
            plot_name = plot_name.replace('of', 'in')
            label = plot_name[0].upper() + plot_name[1:]

            if 'relative' in method_name:
                # Put the relative score as text on the plot for the change.
                massif_name_to_text = {m: ('+' if v > 0 else '') + str(int(v)) + '\%' for m, v in
                                       self.method_name_and_order_to_d(self.moment_names[2], order).items()}
            else:
                # Put the change score as text on the plot for the change.
                massif_name_to_text = {m: ('+' if v > 0 else '') + str(round(v, 1)) for m, v in
                                       self.method_name_and_order_to_d(self.moment_names[1], order).items()}

            # Some prints
            # print(OneFoldFit.COVARIATE_BEFORE_TEMPERATURE, OneFoldFit.COVARIATE_AFTER_TEMPERATURE, 'Order is {}'.format(order))
            s = 'relative change' if 'relative' in method_name else 'absolute change'
            d = self.method_name_and_order_to_d(method_name, order)
            inverse_d = {v: k for k,v in d.items()}
            print(d)
            # print(inverse_d)
            values = list(d.values())
            print(f"mean{np.mean(values)}")
            # print(f"{s} values: min{np.min(values)} max{np.max(values)}")
            # print(f"{s} massif min{inverse_d[np.min(values)]} max{inverse_d[np.max(values)]}")

        if only_print:
            return

        parenthesis = self.study.variable_unit if 'relative' not in method_name else '\%'
        ylabel = label + ' ({})'.format(parenthesis)
        plot_name += 'at +{}$^o$C'.format(OneFoldFit.COVARIATE_AFTER_TEMPERATURE)

        add_colorbar = self.study.altitude in [2100, 3600]

        is_return_level_plot = (self.moment_names.index(method_name) == 0) and (order is None)
        fontsize_label = 13
        if is_return_level_plot:
            print('average return level values:')
            print(np.mean(list(massif_name_to_value.values())))
            if max_abs_change is None:
                max_abs_change = max([v for v in massif_name_to_value.values()])

            cmap = plt.cm.Oranges
            massif_name_to_text = {m: round(v, 1) for m, v in massif_name_to_value.items()}
            if not snowfall:
                if self.altitude_group.altitude == 1500:
                    max_abs_change = 9.9
                else:
                    max_abs_change = None
                graduation = 2
            else:
                graduation = 10
            half_cmap_for_positive = False
        else:
            half_cmap_for_positive = True
            if not snowfall:
                cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][1]
                if 'relative' in method_name:
                    graduation = 10
                    max_abs_change = 60
                else:
                    graduation = 1
                    max_abs_change = 4
            else:
                cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic, plt.cm.BrBG][-1]
                graduation = 5
                max_abs_change = max_abs_change
            cmap = remove_the_extreme_colors(cmap)

        if snowfall:
            negative_and_positive_values = self.moment_names.index(method_name) > 0
        else:
            negative_and_positive_values = False

        fontsize_label = 14
        # Plot the map
        self.plot_map(cmap=cmap, graduation=graduation,
                      label=ylabel, massif_name_to_value=massif_name_to_value,
                      plot_name=plot_name, add_x_label=False,
                      negative_and_positive_values=negative_and_positive_values,
                      altitude=self.altitude_group.reference_altitude,
                      add_colorbar=add_colorbar,
                      max_abs_change=max_abs_change,
                      fontsize_label=fontsize_label,
                      half_cmap_for_positive=half_cmap_for_positive,
                      massif_name_to_text=massif_name_to_text
                      )
        if snowfall:
            return None
        else:
            return max_abs_change

    def plot_qqplots(self):
        qqplots_metrics = []
        massif_name_to_unconstrained_quantile = dict()
        for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
            unconstrained_empirical_quantiles = one_fold_fit.best_estimator.sorted_empirical_standard_gumbel_quantiles()
            massif_name_to_unconstrained_quantile[massif_name] = unconstrained_empirical_quantiles
        # Sort massif names by group
        massif_name_to_max_unconstrained_quantile = {m: max(u) for m, u in massif_name_to_unconstrained_quantile.items()}
        massif_names = list(massif_name_to_max_unconstrained_quantile.keys())
        sorted_massif_names = sorted(massif_names, key=lambda m: massif_name_to_max_unconstrained_quantile[m], reverse=True)
        for i in range(4):
            group_massif_names = sorted_massif_names[6*i:6*(i+1)]
            all_quantiles = []

            print(i, group_massif_names)
            ax = plt.gca()
            for massif_name in group_massif_names:
                unconstrained_empirical_quantiles = massif_name_to_unconstrained_quantile[massif_name]
                one_fold_fit = self.massif_name_to_one_fold_fit[massif_name]
                massif_name_corrected = massif_name.replace('_', ' ')

                altitude = self.altitude_group.reference_altitude

                # We filter on the transformed gumbel quantiles for the altitude of interest
                n = len(unconstrained_empirical_quantiles)
                if n > 0:
                    standard_gumbel_quantiles = one_fold_fit.standard_gumbel_quantiles(n=n)
                    ax.plot(standard_gumbel_quantiles, unconstrained_empirical_quantiles, linestyle='None',
                            label='{} massif'.format(massif_name_corrected, altitude), marker='o')

                    all_quantiles.extend(standard_gumbel_quantiles)
                    all_quantiles.extend(unconstrained_empirical_quantiles)
                    pvalue = get_pvalue_anderson_darling_test(unconstrained_empirical_quantiles)
                    qqplots_metrics.append(pvalue)

                size_label = 20
                ax.set_xlabel("Theoretical quantile", fontsize=size_label)
                ax.set_ylabel("Empirical quantile", fontsize=size_label)

            epsilon = 0.5
            ax_lim = [min(all_quantiles) - epsilon, max(all_quantiles) + epsilon]
            ax.set_xlim(ax_lim)
            ax.set_ylim(ax_lim)

            ax.plot(ax_lim, ax_lim, color='k')
            ticks = [i for i in range(ceil(ax_lim[0]), floor(ax_lim[1]) + 1)]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            labelsize = 15
            ax.tick_params(labelsize=labelsize)
            plot_name = 'qqplot/{}'.format(i)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], prop={'size': 12}, loc="lower right")
            self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show, no_title=True)
            plt.close()

        return qqplots_metrics
