from typing import Dict

import numpy as np
from cached_property import cached_property

from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.trend_analysis.abstract_score import MeanScore
from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_one_parameter import GevScaleTrendTest, \
    GevLocationTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest


class StudyVisualizerForNonStationaryTrends(StudyVisualizer):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 score_class=MeanScore):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, score_class)
        self.non_stationary_trend_test = [GevLocationTrendTest, GevScaleTrendTest, GevLocationAndScaleTrendTest]
        self.non_stationary_trend_test_to_marker = dict(zip(self.non_stationary_trend_test, ["s", "^", "D"]))

    # Utils

    @cached_property
    def massif_name_to_years_and_maxima(self):
        d = {}
        df_maxima = self.study.observations_annual_maxima.df_maxima_gev
        years = np.array(df_maxima.columns)
        for massif_name, s_maxima in df_maxima.iterrows():
            d[massif_name] = (years, np.array(s_maxima))
        return d

    @cached_property
    def massif_name_to_psnow(self):
        return {m: np.count_nonzero(maxima) / len(maxima) for m, (_, maxima) in
                self.massif_name_to_years_and_maxima.items()}

    @cached_property
    def massif_name_to_non_null_years_and_maxima(self):
        d = {}
        for m, (years, maxima) in self.massif_name_to_years_and_maxima.items():
            mask = np.nonzero(maxima)
            d[m] = (years[mask], maxima[mask])
        return d

    @cached_property
    def massif_name_to_minimized_aic_non_stationary_trend_test(self) -> Dict[str, AbstractGevTrendTest]:
        starting_year = 1958
        massif_name_to_trend_test_that_minimized_aic = {}
        for massif_name, (x, y) in self.massif_name_to_non_null_years_and_maxima.items():
            non_stationary_trend_test = [t(x, y, starting_year) for t in self.non_stationary_trend_test]
            trend_test_that_minimized_aic = sorted(non_stationary_trend_test, key=lambda t: t.aic)[0]
            massif_name_to_trend_test_that_minimized_aic[massif_name] = trend_test_that_minimized_aic
        return massif_name_to_trend_test_that_minimized_aic

    # Part 1 - Trends

    def plot_trends(self):
        v = max(abs(min(self.massif_name_to_tdrl_value.values())), max(self.massif_name_to_tdrl_value.values()))
        vmin, vmax = -v, v
        self.study.visualize_study(massif_name_to_value=self.massif_name_to_tdrl_value, vmin=vmin, vmax=vmax,
                                   replace_blue_by_white=False, axis_off=True, show_label=False,
                                   add_colorbar=True,
                                   massif_name_to_marker_style=self.massif_name_to_marker_style)

    @cached_property
    def massif_name_to_tdrl_value(self):
        return {m: t.time_derivative_of_return_level for m, t in
                self.massif_name_to_minimized_aic_non_stationary_trend_test.items()}

    @property
    def massif_name_to_marker_style(self):
        d = {}
        for m, t in self.massif_name_to_minimized_aic_non_stationary_trend_test.items():
            d[m] = {'marker': self.non_stationary_trend_test_to_marker[type(t)],
                    'color': 'k',
                    'markersize': 5,
                    'fillstyle': 'full' if t.is_significant else 'none'}
        return d

    # Part 1 - Uncertainty return level plot

    @property
    def massif_name_to_minimized_aic_model_class(self):
        return {m: t.unconstrained_model_class for m, t in
                self.massif_name_to_minimized_aic_non_stationary_trend_test.items()}

    def massif_name_to_uncertainty(self):
        pass
