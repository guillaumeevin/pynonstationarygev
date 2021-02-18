from typing import Dict, Tuple, List

from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import DefaultAltitudeGroup


class AbstractEnsembleFit(object):

    def __init__(self, massif_names, gcm_rcm_couple_to_altitude_studies: Dict[Tuple[str, str], AltitudesStudies],
                 models_classes,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 only_models_that_pass_goodness_of_fit_test=True,
                 confidence_interval_based_on_delta_method=False,
                 ):
        self.massif_names = massif_names
        self.models_classes = models_classes
        self.gcm_rcm_couple_to_altitude_studies = gcm_rcm_couple_to_altitude_studies
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.only_models_that_pass_goodness_of_fit_test = only_models_that_pass_goodness_of_fit_test
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method


    def plot(self):
        raise NotImplementedError