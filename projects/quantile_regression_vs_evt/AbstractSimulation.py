from typing import Dict, List
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.quantile_estimator.abstract_quantile_estimator import AbstractQuantileEstimator
from extreme_fit.estimator.quantile_estimator.quantile_estimator_from_margin import QuantileEstimatorFromMargin
from extreme_fit.estimator.quantile_estimator.quantile_estimator_from_regression import QuantileRegressionEstimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.quantile_model.quantile_regression_model import AbstractQuantileRegressionModel
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractSimulation(object):

    def __init__(self, nb_time_series, quantile, time_series_lengths=None, multiprocessing=False,
                 model_classes=None, transformation_class=CenteredScaledNormalization):
        self.transformation_class = transformation_class
        self.models_classes = model_classes
        self.multiprocessing = multiprocessing
        self.quantile = quantile
        self.time_series_lengths = time_series_lengths
        self.nb_time_series = nb_time_series

    def generate_all_observation(self, nb_time_series, length) -> List[AbstractSpatioTemporalObservations]:
        raise NotImplementedError

    @cached_property
    def time_series_length_to_observation_list(self) -> Dict[int, List[AbstractSpatioTemporalObservations]]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            d[length] = self.generate_all_observation(self.nb_time_series, length)
        return d

    @cached_property
    def time_series_length_to_coordinates(self) -> Dict[int, AbstractCoordinates]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            d[length] = ConsecutiveTemporalCoordinates.\
                from_nb_temporal_steps(length, transformation_class=self.transformation_class)
        return d

    @cached_property
    def model_class_to_time_series_length_to_estimators(self):
        d = OrderedDict()
        for model_class in self.models_classes:
            d_sub = OrderedDict()
            for time_series_length, observation_list in self.time_series_length_to_observation_list.items():
                coordinates = self.time_series_length_to_coordinates[time_series_length]
                estimators = []
                for observations in observation_list:
                    estimators.append(self.get_fitted_quantile_estimator(model_class, observations, coordinates))
                d_sub[time_series_length] = estimators
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
        for model_class, d_sub in self.model_class_to_time_series_length_to_estimators.items():
            length_to_error_values = OrderedDict()
            for length, estimators_fitted in d_sub.items():
                errors = self.compute_errors(length, estimators_fitted)
                error_values = [np.quantile(errors, q=0.025), np.mean(errors), np.quantile(errors, q=0.975)]
                length_to_error_values[length] = error_values
            d[model_class] = length_to_error_values
        return d

    def compute_errors(self, length: int, estimators_fitted: List[AbstractQuantileEstimator]):
        raise NotImplementedError

    def plot_error_for_last_year_quantile(self, show=True):
        ax = plt.gca()
        for model_class, length_to_error_values in self.model_class_to_error_last_year_quantile.items():
            lengths = list(length_to_error_values.keys())
            errors_values = np.array(list(length_to_error_values.values()))
            mean_error = errors_values[:, 1]
            label = get_display_name_from_object_type(model_class)
            ax.plot(lengths, mean_error, label=label)
            ax.set_xlabel('# Data')
            ax.set_ylabel('Relative error for the {} quantile at the last coordinate'.format(self.quantile))
            ax.legend()
        if show:
            plt.show()
