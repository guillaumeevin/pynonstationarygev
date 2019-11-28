from collections import OrderedDict
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from typing import Dict, Tuple

import numpy as np
from cached_property import cached_property

from experiment.eurocode_data.utils import EUROCODE_QUANTILE, EUROCODE_RETURN_LEVEL_STR
from experiment.meteo_france_data.plot.create_shifted_cmap import get_shifted_map, get_colors
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.paper_past_snow_loads.check_mcmc_convergence_for_return_levels.gelman_convergence_test import \
    compute_gelman_convergence_value
from experiment.trend_analysis.abstract_score import MeanScore
from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_one_parameter import GevScaleTrendTest, \
    GevLocationTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    compute_eurocode_confidence_interval, EurocodeConfidenceIntervalFromExtremes
from root_utils import NB_CORES


class StudyVisualizerForNonStationaryTrends(StudyVisualizer):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 score_class=MeanScore,
                 uncertainty_methods=None,
                 non_stationary_contexts=None,
                 uncertainty_massif_names=None,
                 effective_temporal_covariate=2017):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, score_class)
        # Add some attributes
        self.effective_temporal_covariate = effective_temporal_covariate
        self.non_stationary_contexts = non_stationary_contexts
        self.uncertainty_methods = uncertainty_methods
        self.uncertainty_massif_names = uncertainty_massif_names
        # Assign some default arguments
        if self.non_stationary_contexts is None:
            self.non_stationary_contexts = [False, True][:1]
        if self.uncertainty_methods is None:
            self.uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                                        ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
        if self.uncertainty_massif_names is None:
            self.uncertainty_massif_names = self.study.study_massif_names
        # Assign default argument for the non stationary trends
        self.non_stationary_trend_test = [GevLocationTrendTest, GevScaleTrendTest, GevLocationAndScaleTrendTest]
        self.non_stationary_trend_test_to_marker = dict(zip(self.non_stationary_trend_test, ["s", "^", "D"]))

        self.global_max_abs_tdrl = None

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
    def massif_name_to_eurocode_quantile_level_in_practice(self):
        """Due to missing data, the the eurocode quantile which 0.98 if we have all the data
        correspond in practice to the quantile psnow x 0.98 of the data where there is snow"""
        return {m: 1 - ((1 - EUROCODE_QUANTILE) / p_snow) for m, p_snow in self.massif_name_to_psnow.items()}

    @cached_property
    def massif_name_to_non_null_years_and_maxima(self):
        d = {}
        for m, (years, maxima) in self.massif_name_to_years_and_maxima.items():
            mask = np.nonzero(maxima)
            d[m] = (years[mask], maxima[mask])
        return d

    @cached_property
    def massif_name_to_minimized_aic_non_stationary_trend_test(self) -> Dict[str, AbstractGevTrendTest]:
        starting_year = None
        massif_name_to_trend_test_that_minimized_aic = {}
        for massif_name, (x, y) in self.massif_name_to_non_null_years_and_maxima.items():
            quantile_level = self.massif_name_to_eurocode_quantile_level_in_practice[massif_name]
            non_stationary_trend_test = [
                t(years=x, maxima=y, starting_year=starting_year, quantile_level=quantile_level)
                for t in self.non_stationary_trend_test]
            trend_test_that_minimized_aic = sorted(non_stationary_trend_test, key=lambda t: t.aic)[0]
            massif_name_to_trend_test_that_minimized_aic[massif_name] = trend_test_that_minimized_aic
        return massif_name_to_trend_test_that_minimized_aic

    # Part 1 - Trends

    @property
    def max_abs_tdrl(self):
        return max(abs(min(self.massif_name_to_tdrl_value.values())), max(self.massif_name_to_tdrl_value.values()))

    @cached_property
    def _max_abs_tdrl(self):
        return self.global_max_abs_tdrl if self.global_max_abs_tdrl is not None else self.max_abs_tdrl

    def plot_trends(self, max_abs_tdrl=None):
        if max_abs_tdrl is not None:
            self.global_max_abs_tdrl = max_abs_tdrl
        self.study.visualize_study(massif_name_to_value=self.massif_name_to_tdrl_value,
                                   replace_blue_by_white=False, axis_off=False, show_label=False,
                                   add_colorbar=True,
                                   massif_name_to_marker_style=self.massif_name_to_marker_style,
                                   massif_name_to_color=self.massif_name_to_tdrl_color,
                                   cmap=self.cmap,
                                   show=self.show,
                                   ticks_values_and_labels=self.ticks_values_and_labels,
                                   label=self.label_tdrl_bar + ' for {}'.format(EUROCODE_RETURN_LEVEL_STR))
        self.plot_name = 'tdlr_trends'
        self.show_or_save_to_file(add_classic_title=False, tight_layout=False, no_title=True)
        plt.close()

    @property
    def label_tdrl_bar(self):
        return 'Change in {} years'.format(AbstractGevTrendTest.nb_years_for_quantile_evolution)

    @property
    def ticks_values_and_labels(self):
        graduation = 0.2
        positive_ticks = []
        tick = graduation
        while tick < self._max_abs_tdrl:
            positive_ticks.append(round(tick, 1))
            tick += graduation
        all_ticks_labels = [-t for t in positive_ticks] + [0] + positive_ticks
        ticks_values = [((t / self._max_abs_tdrl) + 1) / 2 for t in all_ticks_labels]
        return ticks_values, all_ticks_labels

    @cached_property
    def massif_name_to_tdrl_value(self):
        return {m: t.time_derivative_of_return_level for m, t in
                self.massif_name_to_minimized_aic_non_stationary_trend_test.items()}

    @cached_property
    def cmap(self):
        return get_shifted_map(-self._max_abs_tdrl, self._max_abs_tdrl)

    @cached_property
    def massif_name_to_tdrl_color(self):
        return {m: get_colors([v], self.cmap, -self._max_abs_tdrl, self._max_abs_tdrl)[0]
                for m, v in self.massif_name_to_tdrl_value.items()}

    @cached_property
    def massif_name_to_marker_style(self):
        d = {}
        for m, t in self.massif_name_to_minimized_aic_non_stationary_trend_test.items():
            d[m] = {'marker': self.non_stationary_trend_test_to_marker[type(t)],
                    'color': 'k',
                    'markersize': 5,
                    'fillstyle': 'full' if t.is_significant else 'none'}
        return d

    # Part 2 - Uncertainty return level plot

    def massif_name_and_non_stationary_context_to_model_class(self, massif_name, non_stationary_context):
        if not non_stationary_context:
            return StationaryTemporalModel
        else:
            return self.massif_name_to_minimized_aic_non_stationary_trend_test[massif_name].unconstrained_model_class

    @property
    def nb_contexts(self):
        return len(self.non_stationary_contexts)

    def all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class(self, ci_method, non_stationary_context) \
            -> Dict[str, Tuple[int, EurocodeConfidenceIntervalFromExtremes]]:
        # Compute for the uncertainty massif names
        arguments = [
            [self.massif_name_to_non_null_years_and_maxima[m],
             self.massif_name_and_non_stationary_context_to_model_class(m, non_stationary_context),
             ci_method, self.effective_temporal_covariate,
             self.massif_name_to_eurocode_quantile_level_in_practice[m]
             ] for m in self.uncertainty_massif_names]
        if self.multiprocessing:
            with Pool(NB_CORES) as p:
                res = p.starmap(compute_eurocode_confidence_interval, arguments)
        else:
            res = [compute_eurocode_confidence_interval(*argument) for argument in arguments]
        massif_name_to_eurocode_return_level_uncertainty = dict(zip(self.uncertainty_massif_names, res))
        # For the rest of the massif names. Create a Eurocode Return Level Uncertainty as nan
        for massif_name in set(self.study.all_massif_names) - set(self.uncertainty_massif_names):
            massif_name_to_eurocode_return_level_uncertainty[massif_name] = self.default_eurocode_uncertainty
        return massif_name_to_eurocode_return_level_uncertainty

    @cached_property
    def default_eurocode_uncertainty(self):
        return EurocodeConfidenceIntervalFromExtremes(mean_estimate=np.nan, confidence_interval=(np.nan, np.nan))

    @cached_property
    def triplet_to_eurocode_uncertainty(self):
        d = {}
        for ci_method in self.uncertainty_methods:
            for non_stationary_uncertainty in self.non_stationary_contexts:
                for massif_name, eurocode_uncertainty in self.all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class(
                        ci_method, non_stationary_uncertainty).items():
                    d[(ci_method, non_stationary_uncertainty, massif_name)] = eurocode_uncertainty
        return d

    def model_name_to_uncertainty_method_to_ratio_above_eurocode(self):
        assert self.uncertainty_massif_names == self.study.study_massif_names

    # Some checks with Gelman convergence diagnosis

    def massif_name_to_gelman_convergence_value(self, mcmc_iterations, model_class, nb_chains):
        arguments = [(self.massif_name_to_non_null_years_and_maxima[m], mcmc_iterations, model_class, nb_chains)
                     for m in self.uncertainty_massif_names]
        if self.multiprocessing:
            with Pool(NB_CORES) as p:
                res = p.starmap(compute_gelman_convergence_value, arguments)
        else:
            res = [compute_gelman_convergence_value(*argument) for argument in arguments]
        return dict(zip(self.uncertainty_massif_names, res))
