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
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues
from projects.exceeding_snow_loads.utils import NON_STATIONARY_TREND_TEST_PAPER_2


class StudyVisualizerForMeanValuesWithMeanAic(StudyVisualizerForMeanValues):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 uncertainty_methods=None, model_subsets_for_uncertainty=None, uncertainty_massif_names=None,
                 effective_temporal_covariate=YEAR_OF_INTEREST_FOR_RETURN_LEVEL, relative_change_trend_plot=True,
                 non_stationary_trend_test_to_marker=None, fit_method=MarginFitMethod.extremes_fevd_mle,
                 select_only_acceptable_shape_parameter=True, fit_gev_only_on_non_null_maxima=False,
                 fit_only_time_series_with_ninety_percent_of_non_null_values=True):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, uncertainty_methods, model_subsets_for_uncertainty,
                         uncertainty_massif_names, effective_temporal_covariate, relative_change_trend_plot,
                         non_stationary_trend_test_to_marker, fit_method, select_only_acceptable_shape_parameter,
                         fit_gev_only_on_non_null_maxima, fit_only_time_series_with_ninety_percent_of_non_null_values)
        self.massif_name_to_trend_test_with_minimial_mean_aic = None

    @property
    def massif_name_to_trend_test_that_minimized_aic(self) -> Dict[str, AbstractGevTrendTest]:
        if self.massif_name_to_trend_test_with_minimial_mean_aic is None:
            raise NotImplementedError('Aggregation must be run first')
        else:
            return self.massif_name_to_trend_test_with_minimial_mean_aic

    @property
    def massif_names(self):
        return [m for m, _ in self.massif_name_and_trend_test_class_to_trend_test.keys()]

    @cached_property
    def massif_name_and_trend_test_class_to_trend_test(self):
        d = {}
        for massif_name in self.massif_name_to_years_and_maxima_for_model_fitting.keys():
            trend_tests = self.get_sorted_trend_test(massif_name)
            for trend_test in trend_tests:
                d[(massif_name, type(trend_test))] = trend_test
        return d

