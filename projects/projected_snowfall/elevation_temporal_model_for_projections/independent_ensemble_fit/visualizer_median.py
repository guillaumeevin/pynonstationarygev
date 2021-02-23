import copy
from typing import Dict, List

from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit
from projects.projected_snowfall.elevation_temporal_model_for_projections.independent_ensemble_fit.one_fold_fit_median import \
    OneFoldFitMedian


class VisualizerMedian(AltitudesStudiesVisualizerForNonStationaryModels):

    def __init__(self, visualizers: List[AltitudesStudiesVisualizerForNonStationaryModels],
                 model_classes,
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_anderson_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False):
        self.visualizers = visualizers
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
            one_fold_fit_merge = OneFoldFitMedian(one_fold_fit_list, massif_name,
                                                  type(self.altitude_group), self.temporal_covariate_for_fit)
            self._massif_name_to_one_fold_fit[massif_name] = one_fold_fit_merge

    @property
    def massif_name_to_one_fold_fit(self) -> Dict[str, OneFoldFit]:
        return self._massif_name_to_one_fold_fit
