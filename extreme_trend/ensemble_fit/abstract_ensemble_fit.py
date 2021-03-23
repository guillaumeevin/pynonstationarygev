from typing import Dict, Tuple

from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


class AbstractEnsembleFit(object):
    Median_merge = 'Median'
    Mean_merge = 'Mean'
    Together_merge = 'Together'

    def __init__(self, massif_names, gcm_rcm_couple_to_altitude_studies: Dict[Tuple[str, str], AltitudesStudies],
                 models_classes,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 only_models_that_pass_goodness_of_fit_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 ):
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.massif_names = massif_names
        self.models_classes = models_classes
        self.gcm_rcm_couple_to_altitude_studies = gcm_rcm_couple_to_altitude_studies
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.only_models_that_pass_goodness_of_fit_test = only_models_that_pass_goodness_of_fit_test
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method

        # Set appropriate setting
        OneFoldFit.last_year = 2100
        OneFoldFit.nb_years = OneFoldFit.last_year - 2005

    @property
    def altitudes(self):
        raise self.visualizer_list.studies.altitudes

    @property
    def visualizer_list(self):
        raise NotImplementedError