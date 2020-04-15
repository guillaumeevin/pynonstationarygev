import pandas as pd
from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AltitudesStudies(object):

    def __init__(self, study_class, altitudes, transformation_class=None, **kwargs_study):
        self.transformation_class = transformation_class
        self.altitudes = altitudes
        self.altitude_to_study = OrderedDict()
        for altitude in self.altitudes:
            study = study_class(altitude=altitude, **kwargs_study)
            self.altitude_to_study[altitude] = study

    @cached_property
    def study(self) -> AbstractStudy:
        return list(self.altitude_to_study.values())[0]

    def get_dataset(self, massif_name, slicer) -> AbstractDataset:
        pass

    # Coordinates Loader

    @cached_property
    def temporal_coordinates(self):
        return ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=self.study.nb_years,
                                                                     start=self.study.year_min,
                                                                     transformation_class=self.transformation_class)

    @cached_property
    def spatial_coordinates(self):
        return AbstractSpatialCoordinates.from_list_x_coordinates(x_coordinates=self.altitudes,
                                                                  transformation_class=self.transformation_class)

    def random_s_split_temporal(self, train_split_ratio):
        return AbstractSpatioTemporalCoordinates.get_random_s_split_temporal(
            spatial_coordinates=self.spatial_coordinates,
            temporal_coordinates=self.temporal_coordinates,
            train_split_ratio=train_split_ratio)

    def spatio_temporal_coordinates(self, slicer_class: type, s_split_spatial: pd.Series = None,
                                    s_split_temporal: pd.Series = None):
        return AbstractSpatioTemporalCoordinates(slicer_class=slicer_class, s_split_spatial=s_split_spatial,
                                                 s_split_temporal=s_split_temporal,
                                                 transformation_class=self.transformation_class,
                                                 spatial_coordinates=self.spatial_coordinates,
                                                 temporal_coordinates=self.temporal_coordinates)
