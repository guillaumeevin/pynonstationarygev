from typing import List, Dict, Tuple

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.trend_test.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends


class VisualizerNonStationaryEnsemble(AltitudesStudiesVisualizerForNonStationaryModels):

    def __init__(self, gcm_rcm_couple_to_studies: Dict[Tuple[str, str], AltitudesStudies], *args, **kwargs):
        self.gcm_rcm_couple_to_studies = gcm_rcm_couple_to_studies
        studies = list(self.gcm_rcm_couple_to_studies.values())[0]
        super().__init__(studies, *args, **kwargs)

    def get_massif_altitudes(self, massif_name):
        # return self._get_massif_altitudes(massif_name, self.studies)
        raise NotImplementedError

    def get_dataset(self, massif_altitudes, massif_name):
        raise NotImplementedError
        # dataset = self.studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=massif_altitudes)
        # return dataset
