from typing import Union

import pandas as pd

from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
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





