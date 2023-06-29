from typing import Dict, Tuple

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
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

    def get_dataset(self, massif_altitudes, massif_name, gcm_rcm_couple_as_pseudo_truth=None):
        df_coordinates_list = []
        df_maxima_gev_list = []
        for gcm_rcm_couple, studies in self.gcm_rcm_couple_to_studies.items():
            if len(massif_altitudes) == 1:
                dataset = studies.spatio_temporal_dataset_memoize(massif_name, massif_altitudes[0])
            else:
                dataset = studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=massif_altitudes)
            observation_or_pseudo_truth = gcm_rcm_couple in [gcm_rcm_couple_as_pseudo_truth, (None, None)]
            # By default the weight on data is always 1
            weight_on_data = self.weight_on_observation if observation_or_pseudo_truth else 1
            # weight_on_data = 1 if observation_or_pseudo_truth else self.weight_on_observation
            for _ in range(weight_on_data):
                df_coordinates_list.append(dataset.coordinates.df_coordinates(add_climate_informations=True))
                df_maxima_gev_list.append(dataset.observations.df_maxima_gev)

        index = pd.RangeIndex(0, sum([len(df) for df in df_maxima_gev_list]))
        df_maxima_gev = pd.concat(df_maxima_gev_list, axis=0)
        df_maxima_gev.index = index
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df_maxima_gev)
        df = pd.concat(df_coordinates_list, axis=0)
        df.index = index
        coordinates = AbstractCoordinates(df=df)
        coordinates.gcm_rcm_couple_as_pseudo_truth = gcm_rcm_couple_as_pseudo_truth
        dataset = AbstractDataset(observations=observations, coordinates=coordinates)
        return dataset
