import matplotlib
from cached_property import cached_property

from experiment.meteo_france_data.plot.create_shifted_cmap import get_shifted_map
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from experiment.trend_analysis.abstract_score import MeanScore


class StudyVisualizerForShape(StudyVisualizerForNonStationaryTrends):

    def __init__(self, study: AbstractStudy, **kwargs):
        super().__init__(study, **kwargs)

    @cached_property
    def massif_name_to_unconstrained_shape_parameter(self):
        return {m: t.unconstrained_estimator_gev_params.shape
                for m, t in self.massif_name_to_minimized_aic_non_stationary_trend_test.items()}

    @cached_property
    def massif_name_to_change_value(self):
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


