from typing import List

import numpy as np

from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


class OneFoldFitMerge(OneFoldFit):

    def __init__(self, one_fold_fit_list: List[OneFoldFit], massif_name,
                 altitude_group, temporal_covariate_for_fit,
                 first_year, last_year, merge_function=np.median):
        assert len(one_fold_fit_list) > 0
        self.one_fold_fit_list = one_fold_fit_list
        self.altitude_group = altitude_group
        self.massif_name = massif_name
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.merge_function = merge_function
        self.first_year = first_year
        self.last_year = last_year

    def get_moment(self, altitude, temporal_covariate, order=1):
        return self.merge_function([o.get_moment(altitude, temporal_covariate, order) for o in self.one_fold_fit_list])

    def changes_of_moment(self, altitudes, order=1):
        all_changes = [o.changes_of_moment(altitudes, order) for o in self.one_fold_fit_list]
        merged_changes = list(self.merge_function(np.array(all_changes), axis=0))
        assert len(all_changes[0]) == len(merged_changes)
        return merged_changes

    def relative_changes_of_moment(self, altitudes, order=1):
        all_relative_changes = [o.relative_changes_of_moment(altitudes, order) for o in self.one_fold_fit_list]
        merged_relative_changes = list(self.merge_function(np.array(all_relative_changes), axis=0))
        assert len(all_relative_changes[0]) == len(merged_relative_changes)
        return merged_relative_changes

    @property
    def best_shape(self):
        return self.merge_function([o.best_shape for o in self.one_fold_fit_list])
