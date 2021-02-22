from typing import Dict, Tuple, List

from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import DefaultAltitudeGroup
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit
from projects.altitude_spatial_model.altitudes_fit.utils_altitude_studies_visualizer import compute_and_assign_max_abs
from projects.projected_snowfall.elevation_temporal_model_for_projections.ensemble_fit.abstract_ensemble_fit import \
    AbstractEnsembleFit


class IndependentEnsembleFit(AbstractEnsembleFit):
    """For each gcm_rcm_couple, we create a OneFoldFit"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set appropriate setting
        OneFoldFit.last_year = 2100
        OneFoldFit.nb_years = 95
        # Load a classical visualizer
        self.gcm_rcm_couple_to_visualizer = {}
        for gcm_rcm_couple, studies in self.gcm_rcm_couple_to_altitude_studies.items():
            print(gcm_rcm_couple)
            visualizer = AltitudesStudiesVisualizerForNonStationaryModels(studies, self.models_classes,
                                                                          False,
                                                                          self.massif_names, self.fit_method,
                                                                          self.temporal_covariate_for_fit,
                                                                          self.only_models_that_pass_goodness_of_fit_test,
                                                                          self.confidence_interval_based_on_delta_method,
                                                                          self.remove_physically_implausible_models)
            self.gcm_rcm_couple_to_visualizer[gcm_rcm_couple] = visualizer
