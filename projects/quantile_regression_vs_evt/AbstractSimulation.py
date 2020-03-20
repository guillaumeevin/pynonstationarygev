from typing import Dict, List
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.quantile_estimator.quantile_estimator_from_margin import QuantileEstimatorFromMargin
from extreme_fit.estimator.quantile_estimator.quantile_estimator_from_regression import QuantileRegressionEstimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.quantile_model.quantile_regression_model import AbstractQuantileRegressionModel
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class Coordinates(object):
    pass


class AbstractSimulation(object):

    def __init__(self, nb_time_series, quantile, time_series_lengths=None, multiprocessing=False,
                 model_classes=None):
        self.models_classes = model_classes
        self.multiprocessing = multiprocessing
        self.quantile = quantile
        self.time_series_lengths = time_series_lengths
        self.nb_time_series = nb_time_series

    def generate_all_observation(self, nb_time_series, length, coordinates) -> List[AbstractSpatioTemporalObservations]:
        raise NotImplementedError

    @cached_property
    def time_serie_length_to_observation_list(self) -> Dict[int, List[AbstractSpatioTemporalObservations]]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            if self.multiprocessing:
                raise NotImplementedError
            else:
                coordinates = self.time_serie_length_to_coordinates[length]
                d[length] = self.generate_all_observation(self.nb_time_series, length, coordinates)
        return d

    @cached_property
    def time_serie_length_to_coordinates(self) -> Dict[int, Coordinates]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            d[length] = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(length)
        return d

    @cached_property
    def model_class_to_time_serie_length_to_estimator_fitted(self):
        d = OrderedDict()
        for model_class in self.models_classes:
            d_sub = OrderedDict()
            for time_serie_length, observation_list in self.time_serie_length_to_observation_list.items():
                coordinates = self.time_serie_length_to_coordinates[time_serie_length]
                estimators_fitted = []
                for observations in observation_list:
                    estimators_fitted.append(self.get_fitted_quantile_estimator(model_class, observations, coordinates))
                d_sub[time_serie_length] = estimators_fitted
            d[model_class] = d_sub
        return d

    def get_fitted_quantile_estimator(self, model_class, observations, coordinates):
        dataset = AbstractDataset(observations, coordinates)
        if issubclass(model_class, AbstractTemporalLinearMarginModel):
            estimator = QuantileEstimatorFromMargin(dataset, self.quantile, model_class)
        elif issubclass(model_class, AbstractQuantileRegressionModel):
            estimator = QuantileRegressionEstimator(dataset, self.quantile, model_class)
        else:
            raise NotImplementedError
        estimator.fit()
        return estimator

    @cached_property
    def model_class_to_error_last_year_quantile(self):
        d = OrderedDict()
        for model_class, d_sub in self.model_class_to_time_serie_length_to_estimator_fitted.items():
            length_to_error_values = OrderedDict()
            for length, estimators_fitted in d_sub.items():
                errors = []
                # Add the mean, and the quantile of the 95% confidence interval
                error_values = [0.2, 1, 1.3]
                length_to_error_values[length] = error_values
            d[model_class] = length_to_error_values
        print(d)
        return d

    def plot_error_for_last_year_quantile(self, show=True):
        ax = plt.gca()
        for model_class, length_to_error_values in self.model_class_to_error_last_year_quantile.items():
            lengths = list(length_to_error_values.keys())
            errors_values = np.array(list(length_to_error_values.values()))
            print(errors_values.shape)
            mean_error = errors_values[:, 1]
            ax.plot(lengths, mean_error, label=str(model_class))
            ax.legend()
        if show:
            plt.show()
