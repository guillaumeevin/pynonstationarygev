from collections import OrderedDict
from typing import List, Dict

import numpy as np
from cached_property import cached_property

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.quantile_estimator.abstract_quantile_estimator import AbstractQuantileEstimator
from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel
from projects.quantile_regression_vs_evt.AbstractSimulation import AbstractSimulation
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import MarginAnnualMaxima


class GevSimulation(AbstractSimulation):

    @cached_property
    def time_series_lengths_to_margin_model(self) -> Dict[int, AbstractMarginModel]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            coordinates = self.time_serie_length_to_coordinates[length]
            d[length] = self.create_model(coordinates)
        return d

    def create_model(self, coordinates):
        raise NotImplementedError

    def generate_all_observation(self, nb_time_series, length) -> List[AbstractSpatioTemporalObservations]:
        coordinates = self.time_serie_length_to_coordinates[length]
        margin_model = self.time_series_lengths_to_margin_model[length]
        return [MarginAnnualMaxima.from_sampling(nb_obs=length, coordinates=coordinates, margin_model=margin_model)
                for _ in range(nb_time_series)]

    def compute_errors(self, length: int, estimators: List[AbstractQuantileEstimator]):
        coordinates = self.time_serie_length_to_coordinates[length]
        last_coordinate = coordinates.coordinates_values()[-1]
        # Compute true value
        margin_model = self.time_series_lengths_to_margin_model[length]
        true_quantile = margin_model.margin_function_sample.get_gev_params(last_coordinate).quantile(self.quantile)
        # Compute estimated values
        estimated_quantiles = [estimator.function_from_fit.get_quantile(last_coordinate) for estimator in estimators]
        return np.abs(np.array(estimated_quantiles) - true_quantile)


class StationarySimulation(GevSimulation):

    def create_model(self, coordinates):
        gev_param_name_to_coef_list = {
            GevParams.LOC: [0],
            GevParams.SHAPE: [0],
            GevParams.SCALE: [1],
        }
        return StationaryTemporalModel.from_coef_list(coordinates, gev_param_name_to_coef_list,
                                                      fit_method=TemporalMarginFitMethod.extremes_fevd_mle)


class NonStationaryLocationSimulation(GevSimulation):

    def create_model(self, coordinates):
        gev_param_name_to_coef_list = {
            GevParams.LOC: [0, 1],
            GevParams.SHAPE: [0],
            GevParams.SCALE: [1],
        }
        return NonStationaryLocationTemporalModel.from_coef_list(coordinates, gev_param_name_to_coef_list,
                                                                 fit_method=TemporalMarginFitMethod.extremes_fevd_mle)
