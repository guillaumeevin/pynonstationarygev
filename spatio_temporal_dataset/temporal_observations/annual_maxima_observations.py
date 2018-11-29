import pandas as pd

from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.temporal_observations.abstract_temporal_observations import AbstractTemporalObservations


class AnnualMaxima(AbstractTemporalObservations):
    """
    Index are stations index
    Columns are the annual of the maxima
    """
    pass


class MarginAnnualMaxima(AnnualMaxima):

    @classmethod
    def from_sampling(cls, nb_obs: int, coordinates: AbstractCoordinates,
                      margin_model: AbstractMarginModel):
        maxima_gev = margin_model.rmargin_from_nb_obs(nb_obs=nb_obs, coordinates_values=coordinates.coordinates_values)
        df_maxima_gev = pd.DataFrame(data=maxima_gev, index=coordinates.index)
        return cls(df_maxima_gev=df_maxima_gev)


class MaxStableAnnualMaxima(AbstractTemporalObservations):

    @classmethod
    def from_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel, coordinates: AbstractCoordinates):
        maxima_frech = max_stable_model.rmaxstab(nb_obs=nb_obs, coordinates=coordinates.coordinates_values)
        df_maxima_frech = pd.DataFrame(data=maxima_frech, index=coordinates.index)
        return cls(df_maxima_frech=df_maxima_frech)


class FullAnnualMaxima(MaxStableAnnualMaxima):

    @classmethod
    def from_double_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel,
                             coordinates: AbstractCoordinates, margin_model: AbstractMarginModel):
        max_stable_annual_maxima = super().from_sampling(nb_obs, max_stable_model, coordinates)
        #  Compute df_maxima_gev from df_maxima_frech
        maxima_gev = margin_model.rmargin_from_maxima_frech(maxima_frech=max_stable_annual_maxima.maxima_frech(),
                                                            coordinates_values=coordinates.coordinates_values)
        max_stable_annual_maxima.df_maxima_gev = pd.DataFrame(data=maxima_gev, index=coordinates.index)
        return max_stable_annual_maxima
