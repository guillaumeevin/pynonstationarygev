from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property

from extreme_data.eurocode_data.utils import YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import StudyVisualizerForNonStationaryTrends
from projects.exceeding_snow_loads.utils import NON_STATIONARY_TREND_TEST_PAPER_2


class StudyVisualizerForMeanValues(StudyVisualizerForNonStationaryTrends):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 uncertainty_methods=None, model_subsets_for_uncertainty=None, uncertainty_massif_names=None,
                 effective_temporal_covariate=YEAR_OF_INTEREST_FOR_RETURN_LEVEL, relative_change_trend_plot=True,
                 non_stationary_trend_test_to_marker=None, fit_method=MarginFitMethod.extremes_fevd_mle,
                 select_only_acceptable_shape_parameter=True, fit_gev_only_on_non_null_maxima=False,
                 fit_only_time_series_with_ninety_percent_of_non_null_values=True):

        # Change the type of models used in the study
        non_stationary_trend_test_to_marker = {c: '' for c in NON_STATIONARY_TREND_TEST_PAPER_2}

        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, uncertainty_methods, model_subsets_for_uncertainty,
                         uncertainty_massif_names, effective_temporal_covariate, relative_change_trend_plot,
                         non_stationary_trend_test_to_marker, fit_method, select_only_acceptable_shape_parameter,
                         fit_gev_only_on_non_null_maxima, fit_only_time_series_with_ninety_percent_of_non_null_values)
        assert len(self.non_stationary_trend_test) == len(NON_STATIONARY_TREND_TEST_PAPER_2)

    def plot_abstract_fast(self, massif_name_to_value, label, graduation=10.0, cmap=plt.cm.coolwarm, add_x_label=True,
                           negative_and_positive_values=True, add_text=False):
        massif_name_to_text = self.massif_name_to_text if add_text else None
        super().plot_abstract(massif_name_to_value, label, label, self.fit_method, graduation, cmap, add_x_label,
                              negative_and_positive_values, massif_name_to_text)

    @property
    def massif_name_to_text(self):
        d = {}
        for m, t in self.massif_name_to_trend_test_that_minimized_aic.items():
            if t.is_significant:
                name = '$\\textbf{'
                name += t.name
                name += '}$'
            else:
                name = '${}$'.format(t.name)
            d[m] = name
        return d

    # Override the main dict massif_name_to_trend_test_that_minimized_aic

    @property
    def massif_name_to_trend_test_that_minimized_aic(self) -> Dict[str, AbstractGevTrendTest]:
        return {m: t for m, t in super().massif_name_to_trend_test_that_minimized_aic.items()
                if t.goodness_of_fit_anderson_test}

    @property
    def super_massif_name_to_trend_test_that_minimized_aic(self):
        return super().massif_name_to_trend_test_that_minimized_aic

    # Study of the mean

    @cached_property
    def massif_name_to_empirical_mean(self):
        massif_name_to_empirical_value = {}
        for massif_name, maxima in self.study.massif_name_to_annual_maxima.items():
            massif_name_to_empirical_value[massif_name] = np.mean(maxima)
        return massif_name_to_empirical_value

    @cached_property
    def massif_name_to_model_mean_last_year(self):
        massif_name_to_model_value_last_year = {}
        for massif_name, trend_test in self.massif_name_to_trend_test_that_minimized_aic.items():
            massif_name_to_model_value_last_year[massif_name] = trend_test.unconstrained_estimator_gev_params_last_year.mean
        return massif_name_to_model_value_last_year

    @cached_property
    def massif_name_to_relative_difference_for_mean(self):
        massif_name_to_relative_difference = {}
        for massif_name in self.massif_name_to_trend_test_that_minimized_aic.keys():
            e = self.massif_name_to_empirical_mean[massif_name]
            m = self.massif_name_to_model_mean_last_year[massif_name]
            relative_diference = 100 * (m - e) / e
            massif_name_to_relative_difference[massif_name] = relative_diference
        return massif_name_to_relative_difference

    # Study of the change in the mean

    @cached_property
    def massif_name_to_change_ratio_in_empirical_mean(self):
        massif_name_to_empirical_value = {}
        for massif_name, maxima in self.study.massif_name_to_annual_maxima.items():
            index = self.study.ordered_years.index(1989)
            maxima_before, maxima_after = maxima[:index + 1], maxima[index + 1:]
            massif_name_to_empirical_value[massif_name] = np.mean(maxima_after) / np.mean(maxima_before)
        return massif_name_to_empirical_value

    @cached_property
    def massif_name_to_change_ratio_in_model_mean(self):
        massif_name_to_parameter_value = {}
        for massif_name, trend_test in self.massif_name_to_trend_test_that_minimized_aic.items():
            model_mean_before = trend_test.unconstrained_average_mean_value(year_min=self.study.year_min, year_max=1989)
            model_mean_after = trend_test.unconstrained_average_mean_value(year_min=1990, year_max=self.study.year_max)
            massif_name_to_parameter_value[massif_name] = model_mean_after / model_mean_before
        return massif_name_to_parameter_value

    @cached_property
    def massif_name_to_relative_difference_for_change_ratio_in_mean(self):
        massif_name_to_relative_difference = {}
        for massif_name in self.massif_name_to_trend_test_that_minimized_aic.keys():
            e = self.massif_name_to_change_ratio_in_empirical_mean[massif_name]
            m = self.massif_name_to_change_ratio_in_model_mean[massif_name]
            relative_diference = 100 * (m - e) / e
            massif_name_to_relative_difference[massif_name] = relative_diference
        return massif_name_to_relative_difference
