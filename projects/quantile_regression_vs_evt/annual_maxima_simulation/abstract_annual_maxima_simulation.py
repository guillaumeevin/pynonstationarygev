from collections import OrderedDict
from typing import List, Dict

import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.quantile_estimator.abstract_quantile_estimator import AbstractQuantileEstimator
from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from projects.quantile_regression_vs_evt.AbstractSimulation import AbstractSimulation
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AnnualMaximaSimulation(AbstractSimulation):

    @property
    def quantile_estimator(self):
        return self._quantile

    @property
    def quantile_data(self):
        return self._quantile

    @property
    def observations_class(self):
        raise NotImplementedError

    @cached_property
    def time_series_lengths_to_margin_model(self) -> Dict[int, AbstractMarginModel]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            coordinates = self.time_series_length_to_coordinates[length]
            d[length] = self.create_model(coordinates)
        return d

    def create_model(self, coordinates):
        raise NotImplementedError

    def generate_all_observation(self, nb_time_series, length) -> List[AbstractSpatioTemporalObservations]:
        coordinates = self.time_series_length_to_coordinates[length]
        margin_model = self.time_series_lengths_to_margin_model[length]
        return [self.observations_class.from_sampling(nb_obs=1, coordinates=coordinates, margin_model=margin_model)
                for _ in range(nb_time_series)]

    def compute_errors(self, length: int, estimators: List[AbstractQuantileEstimator]):
        coordinates = self.time_series_length_to_coordinates[length]
        last_coordinate = coordinates.coordinates_values()[-1]
        # Compute true value
        margin_model = self.time_series_lengths_to_margin_model[length]
        true_gev_params = margin_model.margin_function_sample.get_gev_params(last_coordinate)
        print(true_gev_params)
        true_quantile = true_gev_params.quantile(self.quantile_data)
        print(self.quantile_data)
        print(true_quantile)
        # Compute estimated values
        estimated_quantiles = [estimator.function_from_fit.get_quantile(last_coordinate) for estimator in estimators]
        return 100 * np.abs(np.array(estimated_quantiles) - true_quantile) / true_quantile
