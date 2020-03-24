import pandas as pd

from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class DailyObservations(AbstractSpatioTemporalObservations):

    def transform_to_standard_shape(self, coordinates: AbstractCoordinates):
        coordinates.df_all_coordinates = pd.concat([coordinates.df_all_coordinates for _ in range(self.nb_obs)])
        df = pd.DataFrame(pd.concat([self.df_maxima_gev[c] for c in self.columns]), index=coordinates.index)
        observation = AbstractSpatioTemporalObservations(df_maxima_gev=df)
        return observation, coordinates


class DailyExp(DailyObservations):

    @classmethod
    def from_sampling(cls, nb_obs: int, coordinates: AbstractCoordinates,
                      margin_model: AbstractMarginModel):
        exponential_values = margin_model.rmargin_from_nb_obs(nb_obs=nb_obs,
                                                              coordinates_values=coordinates.coordinates_values(),
                                                              sample_r_function='rexp')
        df_exponential_values = pd.DataFrame(data=exponential_values, index=coordinates.index)
        return cls(df_maxima_gev=df_exponential_values)
