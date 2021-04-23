import copy
import os
import os.path as op
from typing import List, Dict

import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractDataset(object):

    def __init__(self, observations: AbstractSpatioTemporalObservations, coordinates: AbstractCoordinates, ):
        assert pd.Index.equals(observations.index, coordinates.index), '\n{}\n{}'.format(observations.index,
                                                                                         coordinates.index)
        self.observations = observations  # type: AbstractSpatioTemporalObservations
        self.coordinates = coordinates  # type: AbstractCoordinates

    @classmethod
    def remove_zeros(cls, observations: AbstractSpatioTemporalObservations,
                     coordinates: AbstractCoordinates):
        ind_without_zero = ~(observations.df_maxima_gev == 0).any(axis=1)
        return cls.create_new_dataset(coordinates, ind_without_zero, observations)

    @classmethod
    def create_new_dataset(cls, coordinates, ind, observations):
        # Create new observations
        new_df_maxima_gev = observations.df_maxima_gev.loc[ind].copy()
        new_observations = AbstractSpatioTemporalObservations(df_maxima_gev=new_df_maxima_gev)
        # Create new coordinates
        coordinate_class = type(coordinates)
        new_df = coordinates.df_all_coordinates.loc[ind].copy()
        new_coordinates = coordinate_class(df=new_df)
        return cls(new_observations, new_coordinates)

    @classmethod
    def remove_top_maxima(cls, observations: AbstractSpatioTemporalObservations,
                          coordinates: AbstractSpatioTemporalCoordinates):
        """ We remove the top maxima w.r.t. each spatial coordinates"""
        assert isinstance(coordinates, AbstractSpatioTemporalCoordinates)
        idxs_top = []
        for spatial_coordinate in coordinates.spatial_coordinates.coordinates_values():
            ind = coordinates.df_all_coordinates[coordinates.COORDINATE_X] == spatial_coordinate[0]
            idx = observations.df_maxima_gev.loc[ind].idxmax()[0]
            idxs_top.append(idx)
        ind = ~coordinates.index.isin(idxs_top)
        return cls.create_new_dataset(coordinates, ind, observations)

    @classmethod
    def remove_last_maxima(cls, observations: AbstractSpatioTemporalObservations,
                           coordinates: AbstractSpatioTemporalCoordinates,
                           nb_years=1):
        """ We remove the top maxima w.r.t. each spatial coordinates"""
        assert isinstance(coordinates, AbstractSpatioTemporalCoordinates)
        years = list(coordinates.temporal_coordinates.coordinates_values()[:, 0])
        last_years = sorted(years)[-nb_years:]
        ind = ~coordinates.df_all_coordinates[coordinates.COORDINATE_T].isin(last_years)
        return cls.create_new_dataset(coordinates, ind, observations)

    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        return self.observations.df_maxima_merged.join(self.coordinates.df_coordinates())

    # Observation wrapper

    @property
    def maxima_gev(self) -> np.ndarray:
        return self.observations.maxima_gev

    @property
    def maxima_frech(self) -> np.ndarray:
        return self.observations.maxima_frech

    def set_maxima_frech(self, maxima_frech_values: np.ndarray):
        self.observations.set_maxima_frech(maxima_frech_values)

    # Observation wrapper for fit function

    def transform_maxima_for_spatial_extreme_package(self, array) -> np.ndarray:
        if self.coordinates.has_spatio_temporal_coordinates:
            nb_obs = self.observations.nb_obs
            nb_stations = len(self.coordinates.df_spatial_coordinates())
            nb_steps = self.coordinates.nb_steps
            # Permute array lines
            time_steps = np.array(range(nb_steps))
            c = [time_steps * nb_stations + i for i in range(nb_stations)]
            permutation = np.concatenate(c)
            array = array[permutation]
            # Reshape array
            shape = (nb_stations, nb_steps * nb_obs)
            array = array.reshape(shape)
        return np.transpose(array)

    @property
    def maxima_gev_for_spatial_extremes_package(self) -> np.ndarray:
        return self.transform_maxima_for_spatial_extreme_package(self.maxima_gev)

    @property
    def maxima_frech_for_spatial_extremes_package(self) -> np.ndarray:
        return self.transform_maxima_for_spatial_extreme_package(self.maxima_frech)

    # Coordinates wrapper

    @property
    def df_coordinates(self) -> pd.DataFrame:
        return self.coordinates.df_coordinates()

    def coordinates_values(self) -> np.ndarray:
        return self.coordinates.coordinates_values()

    # Special methods

    def __str__(self) -> str:
        return 'coordinates:\n{}\nobservations:\n{}'.format(self.coordinates.__str__(), self.observations.__str__())
