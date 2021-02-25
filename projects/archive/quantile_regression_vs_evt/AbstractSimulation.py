from multiprocessing.dummy import Pool
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
from root_utils import get_display_name_from_object_type, NB_CORES
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
        self._quantile = quantile
        self.time_series_lengths = time_series_lengths
        self.nb_time_series = nb_time_series
        self.uncertainty_interval_size = 0.5

    @property
    def quantile_estimator(self):
        raise NotImplementedError

    @property
    def quantile_data(self):
        raise NotImplementedError

    def generate_all_observation(self, nb_time_series, length) -> List[AbstractSpatioTemporalObservations]:
        raise NotImplementedError

    @cached_property
    def time_series_length_to_observation_list(self) -> Dict[int, List[AbstractSpatioTemporalObservations]]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            observation_list = self.generate_all_observation(self.nb_time_series, length)
            d[length] = observation_list
        return d

    @cached_property
    def time_series_length_to_coordinates(self) -> Dict[int, AbstractCoordinates]:
        d = OrderedDict()
        for length in self.time_series_lengths:
            d[length] = ConsecutiveTemporalCoordinates. \
                from_nb_temporal_steps(length, transformation_class=self.transformation_class)
        return d

    @cached_property
    def model_class_to_time_series_length_to_estimators(self):
        d = OrderedDict()
        for i, model_class in enumerate(self.models_classes, 1):
            d_sub = OrderedDict()
            for time_series_length, observation_list in self.time_series_length_to_observation_list.items():
                print(model_class, '{}/{}'.format(i, len(self.models_classes)), time_series_length)
                coordinates = self.time_series_length_to_coordinates[time_series_length]

                arguments = [
                    [model_class, observations, coordinates, self.quantile_estimator]
                    for observations in observation_list
                ]
                if self.multiprocessing:
                    raise NotImplementedError('The multiprocessing seems slow compared to the other,'
                                              'maybe it would be best to call an external function rather than'
                                              'a method, but this methods is override in other classes...')
                    # with Pool(NB_CORES) as p:
                    #     estimators = p.starmap(self.get_fitted_quantile_estimator, arguments)
                else:
                    estimators = []
                    for argument in arguments:
                        estimators.append(self.get_fitted_quantile_estimator(*argument))
                d_sub[time_series_length] = estimators
            d[model_class] = d_sub
        return d

    def get_fitted_quantile_estimator(self, model_class, observations, coordinates, quantile_estimator):
        dataset = AbstractDataset(observations, coordinates)
        if issubclass(model_class, AbstractTemporalLinearMarginModel):
            estimator = QuantileEstimatorFromMargin(dataset, quantile_estimator, model_class)
        elif issubclass(model_class, AbstractQuantileRegressionModel):
            estimator = QuantileRegressionEstimator(dataset, quantile_estimator, model_class)
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
                leftover = (1 - self.uncertainty_interval_size) / 2
                error_values = [np.quantile(errors, q=leftover), np.mean(errors), np.quantile(errors, q=1 - leftover)]
                length_to_error_values[length] = error_values
            d[model_class] = length_to_error_values
        return d

    def compute_errors(self, length: int, estimators_fitted: List[AbstractQuantileEstimator]):
        raise NotImplementedError

    def plot_error_for_last_year_quantile(self, show=True):
        # Display properties
        alpha = 0.1
        colors = ['green', 'orange', 'blue', 'red']
        ax = plt.gca()
        for color, (model_class, length_to_error_values) in zip(colors,
                                                                self.model_class_to_error_last_year_quantile.items()):
            lengths = list(length_to_error_values.keys())
            errors_values = np.array(list(length_to_error_values.values()))
            mean_error = errors_values[:, 1]
            label = get_display_name_from_object_type(model_class)
            ax.plot(lengths, mean_error, label=label)
            ax.set_xlabel('# Data')
            ax.set_ylabel(
                'Average (out of {} samples) relative error\nfor the {} estimated '
                'quantile at the last coordinate (%)'.format(
                    self.nb_time_series,
                    self.quantile_estimator))

            lower_bound = errors_values[:, 0]
            upper_bound = errors_values[:, 2]
            # confidence_interval_str = '95 \% confidence interval'
            ax.fill_between(lengths, lower_bound, upper_bound, color=color, alpha=alpha)
            title = "{} + {}".format(get_display_name_from_object_type(type(self)),
                                     get_display_name_from_object_type(self.transformation_class))
            ax.set_title(title)

            ax.legend()
        if show:
            plt.show()