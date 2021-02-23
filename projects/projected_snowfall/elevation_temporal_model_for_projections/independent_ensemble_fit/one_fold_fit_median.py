from typing import List

import numpy as np

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit


class OneFoldFitMedian(OneFoldFit):

    def __init__(self, one_fold_fit_list: List[OneFoldFit], massif_name, altitude_class, temporal_covariate_for_fit):
        self.one_fold_fit_list = one_fold_fit_list
        self.altitude_group = altitude_class()
        self.massif_name = massif_name
        self.temporal_covariate_for_fit = temporal_covariate_for_fit

    def get_moment(self, altitude, temporal_covariate, order=1):
        return np.median([o.get_moment(altitude, temporal_covariate, order) for o in self.one_fold_fit_list])



