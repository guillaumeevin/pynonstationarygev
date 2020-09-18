import copy
import os
import os.path as op
from typing import List, Dict

import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractDataset(object):

    def __init__(self, observations: AbstractSpatioTemporalObservations, coordinates: AbstractCoordinates,):
        assert pd.Index.equals(observations.index, coordinates.index), '\n{}\n{}'.format(observations.index, coordinates.index)
        self.observations = observations  # type: AbstractSpatioTemporalObservations
        self.coordinates = coordinates  # type: AbstractCoordinates

    @classmethod
    def remove_zeros(cls, observations: AbstractSpatioTemporalObservations,
                     coordinates: AbstractCoordinates):
        ind_without_zero = ~(observations.df_maxima_gev == 0).any(axis=1)
        # Create new observations
        new_df_maxima_gev = observations.df_maxima_gev.loc[ind_without_zero].copy()
        new_observations = AbstractSpatioTemporalObservations(df_maxima_gev=new_df_maxima_gev)
        # Create new coordinates
        coordinate_class = type(coordinates)
        new_df = coordinates.df_all_coordinates.loc[ind_without_zero].copy()
        new_coordinates = coordinate_class(df=new_df, slicer_class=type(coordinates.slicer))
        return cls(new_observations, new_coordinates)


    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        return self.observations.df_maxima_merged.join(self.coordinates.df_merged)

    # Observation wrapper

    def maxima_gev(self, split: Split = Split.all) -> np.ndarray:
        return self.observations.maxima_gev(split, self.slicer)

    def maxima_frech(self, split: Split = Split.all) -> np.ndarray:
        return self.observations.maxima_frech(split, self.slicer)

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: Split = Split.all):
        self.observations.set_maxima_frech(maxima_frech_values, split, self.slicer)

    # Observation wrapper for fit function

    def transform_maxima_for_spatial_extreme_package(self, maxima_function, split) -> np.ndarray:
        array = maxima_function(split)
        if self.coordinates.has_spatio_temporal_coordinates:
            nb_obs = self.observations.nb_obs
            nb_stations = self.coordinates.nb_stations(split)
            nb_steps = self.coordinates.nb_steps(split)
            # Permute array lines
            time_steps = np.array(range(nb_steps))
            c = [time_steps * nb_stations + i for i in range(nb_stations)]
            permutation = np.concatenate(c)
            array = array[permutation]
            # Reshape array
            shape = (nb_stations, nb_steps * nb_obs)
            array = array.reshape(shape)
        return np.transpose(array)

    def maxima_gev_for_spatial_extremes_package(self, split: Split = Split.all) -> np.ndarray:
        return self.transform_maxima_for_spatial_extreme_package(self.maxima_gev, split)

    def maxima_frech_for_spatial_extremes_package(self, split: Split = Split.all) -> np.ndarray:
        return self.transform_maxima_for_spatial_extreme_package(self.maxima_frech, split)

    # Coordinates wrapper

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        return self.coordinates.df_coordinates(split=split)

    def coordinates_values(self, split: Split = Split.all) -> np.ndarray:
        return self.coordinates.coordinates_values(split=split)

    # Slicer wrapper

    @property
    def slicer(self) -> AbstractSlicer:
        return self.coordinates.slicer

    # Special methods

    def __str__(self) -> str:
        return 'coordinates:\n{}\nobservations:\n{}'.format(self.coordinates.__str__(), self.observations.__str__())

