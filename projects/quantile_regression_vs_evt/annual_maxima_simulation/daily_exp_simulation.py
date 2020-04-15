from abc import ABC

from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.model.daily_data_model import AbstractModelOnDailyData
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_exp_models import \
    NonStationaryRateTemporalModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from projects.quantile_regression_vs_evt.annual_maxima_simulation.abstract_annual_maxima_simulation import \
    AnnualMaximaSimulation
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import DailyExpAnnualMaxima


class AbstractDailyExpSimulation(AnnualMaximaSimulation, ABC):

    def __init__(self, nb_time_series, quantile, time_series_lengths=None, multiprocessing=False, model_classes=None,
                 transformation_class=CenteredScaledNormalization):
        super().__init__(nb_time_series, quantile, time_series_lengths, multiprocessing, model_classes,
                         transformation_class)

    @property
    def quantile_data(self):
        return 1 - ((1 - self._quantile) / 365)

    @property
    def observations_class(self):
        return DailyExpAnnualMaxima

    def get_fitted_quantile_estimator(self, model_class, observations: DailyExpAnnualMaxima, coordinates,
                                      quantile_estimator):
        if issubclass(model_class, AbstractModelOnDailyData):
            quantile_estimator = self.quantile_data
            observations, coordinates = observations.daily_observations.transform_to_standard_shape(coordinates)
        return super().get_fitted_quantile_estimator(model_class, observations, coordinates, quantile_estimator)


class StationaryExpSimulation(AbstractDailyExpSimulation):

    def create_model(self, coordinates):
        gev_param_name_to_coef_list = {
            AbstractParams.RATE: [10],
        }
        return StationaryTemporalModel.from_coef_list(coordinates, gev_param_name_to_coef_list,
                                                      fit_method=MarginFitMethod.extremes_fevd_mle,
                                                      params_class=ExpParams)


class NonStationaryExpSimulation(AbstractDailyExpSimulation):

    def create_model(self, coordinates):
        gev_param_name_to_coef_list = {
            AbstractParams.RATE: [0.1, 0.01],
        }
        return NonStationaryRateTemporalModel.from_coef_list(coordinates, gev_param_name_to_coef_list,
                                                             fit_method=MarginFitMethod.extremes_fevd_mle,
                                                             params_class=ExpParams)
