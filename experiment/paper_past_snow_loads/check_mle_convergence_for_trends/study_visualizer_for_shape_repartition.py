import matplotlib
from cached_property import cached_property

from experiment.meteo_france_data.plot.create_shifted_cmap import get_shifted_map
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from experiment.trend_analysis.abstract_score import MeanScore
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter import \
    GevStationaryVersusGumbel


class StudyVisualizerForShape(StudyVisualizerForNonStationaryTrends):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 score_class=MeanScore, uncertainty_methods=None, non_stationary_contexts=None,
                 uncertainty_massif_names=None, effective_temporal_covariate=2017, relative_change_trend_plot=True):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, score_class, uncertainty_methods,
                         non_stationary_contexts, uncertainty_massif_names, effective_temporal_covariate,
                         relative_change_trend_plot)

    @cached_property
    def massif_name_to_unconstrained_shape_parameter(self):
        return {m: t.unconstrained_estimator_gev_params.shape
                for m, t in self.massif_name_to_minimized_aic_non_stationary_trend_test.items()}

    @cached_property
    def massif_name_to_change_value(self):
        print(self.massif_name_to_unconstrained_shape_parameter)
        return self.massif_name_to_unconstrained_shape_parameter

    @property
    def label(self):
        return 'Shape parameter value'

    @property
    def graduation(self):
        return 0.1

    @cached_property
    def cmap(self):
        return get_shifted_map(-self._max_abs_change, self._max_abs_change, matplotlib.cm.get_cmap('BrBG_r'))


class StudyVisualizerGumbel(StudyVisualizerForShape):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 score_class=MeanScore, uncertainty_methods=None, non_stationary_contexts=None,
                 uncertainty_massif_names=None, effective_temporal_covariate=2017, relative_change_trend_plot=True):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, score_class, uncertainty_methods,
                         non_stationary_contexts, uncertainty_massif_names, effective_temporal_covariate,
                         relative_change_trend_plot)
        # Assign default argument for the non stationary trends
        self.non_stationary_trend_test = [GevStationaryVersusGumbel]
        self.non_stationary_trend_test_to_marker = dict(zip(self.non_stationary_trend_test, ["o"]))


class StudyVisualizerAll(StudyVisualizerForShape):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False, normalization_under_one_observations=True,
                 score_class=MeanScore, uncertainty_methods=None, non_stationary_contexts=None,
                 uncertainty_massif_names=None, effective_temporal_covariate=2017, relative_change_trend_plot=True):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis,
                         normalization_under_one_observations, score_class, uncertainty_methods,
                         non_stationary_contexts, uncertainty_massif_names, effective_temporal_covariate,
                         relative_change_trend_plot)
