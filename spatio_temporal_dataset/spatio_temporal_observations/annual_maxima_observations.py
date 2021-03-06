from typing import Union

import pandas as pd

from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_fit.model.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations \
    import AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.daily_observations import DailyExp, DailyObservations


class AnnualMaxima(AbstractSpatioTemporalObservations):

    @classmethod
    def from_coordinates(cls, coordinates: AbstractCoordinates, coordinate_values_to_maxima):
        df_coordinates = coordinates.df_all_coordinates
        return cls.from_df_coordinates(coordinate_values_to_maxima, df_coordinates)

    @classmethod
    def from_df_coordinates(cls, coordinate_values_to_maxima, df_coordinates):
        index_to_maxima = {}
        for i, coordinate_values in df_coordinates.iterrows():
            coordinate_values = tuple(coordinate_values)
            index_to_maxima[i] = coordinate_values_to_maxima[coordinate_values]
        df = pd.DataFrame(index_to_maxima, columns=df_coordinates.index).transpose()
        return cls(df_maxima_gev=df)


class MarginAnnualMaxima(AnnualMaxima):

    @classmethod
    def from_sampling(cls, nb_obs: int, coordinates: AbstractCoordinates,
                      margin_model: AbstractMarginModel):
        maxima_gev = margin_model.rmargin_from_nb_obs(nb_obs=nb_obs,
                                                      coordinates_values=coordinates.coordinates_values(),
                                                      sample_r_function='rgev')
        df_maxima_gev = pd.DataFrame(data=maxima_gev, index=coordinates.index)
        return cls(df_maxima_gev=df_maxima_gev)


class DailyExpAnnualMaxima(AnnualMaxima):

    def __init__(self, df_maxima_gev: pd.DataFrame = None, df_maxima_frech: pd.DataFrame = None,
                 daily_observations: Union[None, DailyObservations] = None):
        super().__init__(df_maxima_gev, df_maxima_frech)
        self.daily_observations = daily_observations

    @classmethod
    def from_sampling(cls, nb_obs: int, coordinates: AbstractCoordinates,
                      margin_model: AbstractMarginModel):
        # todo: to take nb_obs into accoutn i could generate nb_obs * 365 observations
        daily_observations = DailyExp.from_sampling(nb_obs=365, coordinates=coordinates, margin_model=margin_model)
        df_daily_values = daily_observations.df_maxima_gev
        df_maxima_gev = pd.DataFrame({'0': df_daily_values.max(axis=1)}, index=df_daily_values.index)
        return cls(df_maxima_gev=df_maxima_gev, daily_observations=daily_observations)


class MaxStableAnnualMaxima(AnnualMaxima):

    @classmethod
    def from_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel, coordinates: AbstractCoordinates,
                      use_rmaxstab_with_2_coordinates=False):
        maxima_frech = max_stable_model.rmaxstab(nb_obs=nb_obs, coordinates_values=coordinates.coordinates_values(),
                                                 use_rmaxstab_with_2_coordinates=use_rmaxstab_with_2_coordinates)
        df_maxima_frech = pd.DataFrame(data=maxima_frech, index=coordinates.index)
        return cls(df_maxima_frech=df_maxima_frech)


class FullAnnualMaxima(MaxStableAnnualMaxima):

    @classmethod
    def from_double_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel,
                             coordinates: AbstractCoordinates, margin_model: AbstractMarginModel):
        max_stable_annual_maxima = super().from_sampling(nb_obs, max_stable_model, coordinates)
        #  Compute df_maxima_gev from df_maxima_frech
        maxima_gev = margin_model.rmargin_from_maxima_frech(maxima_frech=max_stable_annual_maxima.maxima_frech,
                                                            coordinates_values=coordinates.coordinates_values())
        max_stable_annual_maxima.df_maxima_gev = pd.DataFrame(data=maxima_gev, index=coordinates.index)
        return max_stable_annual_maxima


class FullSpatioTemporalAnnualMaxima(MaxStableAnnualMaxima):

    @classmethod
    def from_double_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel,
                             coordinates: AbstractSpatioTemporalCoordinates, margin_model: AbstractMarginModel):
        # Sample with the max stable spatially
        spatial_coordinate = coordinates.spatial_coordinates
        nb_total_obs = nb_obs * coordinates.nb_steps
        max_stable_annual_maxima = super().from_sampling(nb_total_obs, max_stable_model, spatial_coordinate)
        # Convert observation to a spatio temporal index
        max_stable_annual_maxima.convert_to_spatio_temporal_index(coordinates)
        #  Compute df_maxima_gev from df_maxima_frech
        maxima_gev = margin_model.rmargin_from_maxima_frech(maxima_frech=max_stable_annual_maxima.maxima_frech,
                                                            coordinates_values=coordinates.coordinates_values())
        max_stable_annual_maxima.df_maxima_gev = pd.DataFrame(data=maxima_gev, index=coordinates.index)
        return max_stable_annual_maxima
