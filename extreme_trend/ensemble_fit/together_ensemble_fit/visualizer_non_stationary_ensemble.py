from typing import List, Dict, Tuple
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.trend_test.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class VisualizerNonStationaryEnsemble(AltitudesStudiesVisualizerForNonStationaryModels):

    def __init__(self, gcm_rcm_couple_to_studies: Dict[Tuple[str, str], AltitudesStudies], *args, **kwargs):
        self.gcm_rcm_couple_to_studies = gcm_rcm_couple_to_studies
        studies = list(self.gcm_rcm_couple_to_studies.values())[0]
        super().__init__(studies, *args, **kwargs)

    def get_massif_altitudes(self, massif_name):
        altitudes_before_intersection = []
        for studies in self.gcm_rcm_couple_to_studies.values():
            massif_altitudes = self._get_massif_altitudes(massif_name, studies)
            altitudes_before_intersection.append(set(massif_altitudes))
        altitudes_after_intersection = altitudes_before_intersection[0].intersection(*altitudes_before_intersection[1:])
        altitudes_after_intersection = sorted(list(altitudes_after_intersection))
        return altitudes_after_intersection

    def get_dataset(self, massif_altitudes, massif_name):
        df_coordinates_list = []
        df_maxima_gev_list = []
        for studies in self.gcm_rcm_couple_to_studies.values():
            dataset = studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=massif_altitudes)
            df_coordinates_list.append(dataset.coordinates.df_coordinates(add_climate_informations=True))
            df_maxima_gev_list.append(dataset.observations.df_maxima_gev)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=pd.concat(df_maxima_gev_list, axis=0))
        coordinates = AbstractCoordinates(df=pd.concat(df_coordinates_list, axis=0),
                                          slicer_class=type(dataset.slicer))
        dataset = AbstractDataset(observations=observations, coordinates=coordinates)
        return dataset
