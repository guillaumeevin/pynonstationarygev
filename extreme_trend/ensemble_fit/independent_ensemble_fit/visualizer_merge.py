from typing import Dict, List

import numpy as np

from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.independent_ensemble_fit.one_fold_fit_merge import OneFoldFitMerge
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


class VisualizerMerge(AltitudesStudiesVisualizerForNonStationaryModels):

    def __init__(self, visualizers: List[AltitudesStudiesVisualizerForNonStationaryModels],
                 model_classes,
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_anderson_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 merge_function=np.median):
        self.merge_function = merge_function
        self.visualizers = visualizers
        assert len(visualizers) > 0
        super().__init__(studies=visualizers[0].studies, model_classes=model_classes, show=show, massif_names=massif_names,
                         fit_method=fit_method, temporal_covariate_for_fit=temporal_covariate_for_fit,
                         display_only_model_that_pass_anderson_test=display_only_model_that_pass_anderson_test,
                         confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                         remove_physically_implausible_models=remove_physically_implausible_models)

    def load_one_fold_fit(self):
        self._massif_name_to_one_fold_fit = {}
        for massif_name in self.massif_names:
            one_fold_fit_list = [v.massif_name_to_one_fold_fit[massif_name] for v in self.visualizers
                                 if massif_name in v.massif_name_to_one_fold_fit]
            if len(one_fold_fit_list) > 0:
                one_fold_fit_merge = OneFoldFitMerge(one_fold_fit_list, massif_name,
                                                     type(self.altitude_group), self.temporal_covariate_for_fit)
                self._massif_name_to_one_fold_fit[massif_name] = one_fold_fit_merge

    @property
    def massif_name_to_one_fold_fit(self) -> Dict[str, OneFoldFit]:
        return self._massif_name_to_one_fold_fit
