import pandas as pd
from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima


class AltitudesStudies(object):

    def __init__(self, study_class, altitudes,
                 spatial_transformation_class=None, temporal_transformation_class=None,
                 **kwargs_study):
        self.spatial_transformation_class = spatial_transformation_class
        self.temporal_transformation_class = temporal_transformation_class
        self.altitudes = altitudes
        self.altitude_to_study = OrderedDict() # type: OrderedDict[int, AbstractStudy]
        for altitude in self.altitudes:
            study = study_class(altitude=altitude, **kwargs_study)
            self.altitude_to_study[altitude] = study

    @cached_property
    def study(self) -> AbstractStudy:
        return list(self.altitude_to_study.values())[0]

    # Dataset Loader

    def spatio_temporal_dataset(self, massif_name, s_split_spatial: pd.Series = None, s_split_temporal: pd.Series = None):
        coordinates = self.spatio_temporal_coordinates(s_split_spatial, s_split_temporal)
        coordinate_values_to_maxima = {}
        for altitude in self.altitudes:
            study = self.altitude_to_study[altitude]
            for year, maxima in zip(study.ordered_years, study.massif_name_to_annual_maxima[massif_name]):
                coordinate_values_to_maxima[(altitude, year)] = [maxima]
        observations = AnnualMaxima.from_coordinates(coordinates, coordinate_values_to_maxima)
        return AbstractDataset(observations=observations, coordinates=coordinates)

    # Coordinates Loader

    def spatio_temporal_coordinates(self, s_split_spatial: pd.Series = None, s_split_temporal: pd.Series = None):
        slicer_class = AbstractCoordinates.slicer_class_from_s_splits(s_split_spatial=s_split_spatial,
                                                                      s_split_temporal=s_split_temporal)
        return AbstractSpatioTemporalCoordinates(slicer_class=slicer_class,
                                                 s_split_spatial=s_split_spatial,
                                                 s_split_temporal=s_split_temporal,
                                                 transformation_class=self.spatial_transformation_class,
                                                 spatial_coordinates=self.spatial_coordinates,
                                                 temporal_coordinates=self.temporal_coordinates)

    @cached_property
    def temporal_coordinates(self):
        return ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=self.study.nb_years,
                                                                     start=self.study.year_min,
                                                                     transformation_class=self.spatial_transformation_class)

    @cached_property
    def spatial_coordinates(self):
        return AbstractSpatialCoordinates.from_list_x_coordinates(x_coordinates=self.altitudes,
                                                                  transformation_class=self.temporal_transformation_class)

    @cached_property
    def _df_coordinates(self):
        return AbstractSpatioTemporalCoordinates.get_df_from_spatial_and_temporal_coordinates(self.spatial_coordinates,
                                                                                              self.temporal_coordinates)

    def random_s_split_spatial(self, train_split_ratio):
        return AbstractCoordinates.spatial_s_split_from_df(self._df_coordinates, train_split_ratio)

    def random_s_split_temporal(self, train_split_ratio):
        return AbstractCoordinates.temporal_s_split_from_df(self._df_coordinates, train_split_ratio)
