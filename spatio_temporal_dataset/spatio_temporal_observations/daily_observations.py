import pandas as pd

from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class DailyObservations(AbstractSpatioTemporalObservations):
    pass


class DailyExp(AbstractSpatioTemporalObservations):

    @classmethod
    def from_sampling(cls, nb_obs: int, coordinates: AbstractCoordinates,
                      margin_model: AbstractMarginModel):
        exponential_values = margin_model.rmargin_from_nb_obs(nb_obs=nb_obs,
                                                              coordinates_values=coordinates.coordinates_values())
        df_exponential_values = pd.DataFrame(data=exponential_values, index=coordinates.index)
        return cls(df_maxima_gev=df_exponential_values)
