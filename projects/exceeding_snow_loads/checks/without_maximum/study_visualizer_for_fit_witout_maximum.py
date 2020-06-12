from typing import Dict, Tuple

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends


class StudyVisualizerForFitWithoutMaximum(StudyVisualizerForNonStationaryTrends):

    def __init__(self, study: AbstractStudy, **kwargs):
        super().__init__(study, **kwargs)

    @cached_property
    def massif_name_to_maximum_index_for_non_null_values(self) -> Tuple[Dict, Dict]:
        d = super().massif_name_to_years_and_maxima_for_model_fitting
        d_without_maximum = {}
        d_maximum = {}
        for m, (years, maxima) in d.items():
            idx = np.argmax(maxima)
            maximum = maxima[idx]
            maxima = np.delete(maxima, idx)
            years = np.delete(years, idx)
            d_without_maximum[m] = (years, maxima)
            d_maximum[m] = maximum
        return d_without_maximum, d_maximum

    @property
    def massif_name_to_maximum(self) -> Dict:
        return self.massif_name_to_maximum_index_for_non_null_values[1]

    @cached_property
    def massif_name_to_years_and_maxima_for_model_fitting(self):
        return self.massif_name_to_maximum_index_for_non_null_values[0]

    def maximum_value_test(self):
        diff = []
        for massif_name, maximum in self.massif_name_to_maximum.items():
            t = self.massif_name_to_trend_test_that_minimized_aic[massif_name]
            msg = '{} {}m'.format(massif_name, self.study.altitude)
            upper_bound = t.unconstrained_estimator_gev_params_last_year.density_upper_bound
            if not np.isinf(upper_bound):
                diff.append(upper_bound - maximum)
            assert maximum <= upper_bound, msg
        if len(diff) > 1:
            print('{} mean difference={}'.format(self.study.altitude, min(diff)))










