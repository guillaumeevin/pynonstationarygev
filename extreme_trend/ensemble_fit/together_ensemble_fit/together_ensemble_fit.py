from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble


class TogetherEnsembleFit(AbstractEnsembleFit):
    """We create a single OneFoldFit for all gcm_rcm_couples"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = VisualizerNonStationaryEnsemble(self.gcm_rcm_couple_to_altitude_studies,
                                                     self.models_classes,
                                                     False,
                                                     self.massif_names, self.fit_method,
                                                     self.temporal_covariate_for_fit,
                                                     self.only_models_that_pass_goodness_of_fit_test,
                                                     self.confidence_interval_based_on_delta_method,
                                                     self.remove_physically_implausible_models
                                                     )

    @property
    def visualizer_list(self):
        return [self.visualizer]

