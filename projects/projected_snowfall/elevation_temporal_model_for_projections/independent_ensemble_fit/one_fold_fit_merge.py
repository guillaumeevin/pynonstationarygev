from typing import List

import numpy as np

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit


class OneFoldFitMerge(OneFoldFit):

    def __init__(self, one_fold_fit_list: List[OneFoldFit], massif_name, altitude_class, temporal_covariate_for_fit,
                 merge_function=np.median):
        assert len(one_fold_fit_list) > 0
        self.one_fold_fit_list = one_fold_fit_list
        self.altitude_group = altitude_class()
        self.massif_name = massif_name
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.merge_function = merge_function

    def get_moment(self, altitude, temporal_covariate, order=1):
        return self.merge_function([o.get_moment(altitude, temporal_covariate, order) for o in self.one_fold_fit_list])



