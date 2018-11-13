import pandas as pd

from extreme_estimator.R_fit.gev_fit.abstract_margin_model import AbstractMarginModel
from extreme_estimator.R_fit.max_stable_fit.abstract_max_stable_model import AbstractMaxStableModel
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates
from spatio_temporal_dataset.temporal_observations.abstract_temporal_observations import AbstractTemporalObservations


class AnnualMaxima(AbstractTemporalObservations):
    """
    Index are stations index
    Columns are the annual of the maxima
    """
    pass


class MarginAnnualMaxima(AnnualMaxima):

    @classmethod
    def from_sampling(cls, nb_obs: int, spatial_coordinates: AbstractSpatialCoordinates,
                      margin_model: AbstractMarginModel):
        maxima = margin_model.rmargin(nb_obs=nb_obs, coord=spatial_coordinates.coord)
        df_maxima = pd.DataFrame(data=maxima, index=spatial_coordinates.index)
        return cls(df_maxima=df_maxima)


class MaxStableAnnualMaxima(AbstractTemporalObservations):

    @classmethod
    def from_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel,
                      spatial_coordinates: AbstractSpatialCoordinates):
        maxima_normalized = max_stable_model.rmaxstab(nb_obs=nb_obs, coord=spatial_coordinates.coord)
        df_maxima_normalized = pd.DataFrame(data=maxima_normalized, index=spatial_coordinates.index)
        return cls(df_maxima_normalized=df_maxima_normalized)


class FullAnnualMaxima(MaxStableAnnualMaxima):

    @classmethod
    def from_double_sampling(cls, nb_obs: int, max_stable_model: AbstractMaxStableModel,
                             spatial_coordinates: AbstractSpatialCoordinates,
                             margin_model: AbstractMarginModel):
        max_stable_annual_maxima = super().from_sampling(nb_obs, max_stable_model, spatial_coordinates)
        #  Compute df_maxima from df_maxima_normalized
        maxima = margin_model.get_maxima(max_stable_annual_maxima.maxima_normalized, spatial_coordinates.coord)
        max_stable_annual_maxima.df_maxima = pd.DataFrame(data=maxima, index=spatial_coordinates.index)
        return max_stable_annual_maxima
