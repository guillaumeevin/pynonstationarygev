import time
from multiprocessing import Pool
from typing import Union

import pandas as pd

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from scipy.stats import chi2
from extreme_fit.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin, AbstractFullEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator, \
    AbstractMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import \
    LinearStationaryMarginModel, LinearNonStationaryLocationMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    StationaryTemporalModel, NonStationaryLocationTemporalModel
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.utils import OptimizationConstants
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from root_utils import get_display_name_from_object_type


class AbstractNonStationaryTrendTest(object):
    RESULT_ATTRIBUTE_METRIC = 'deviance'

    def __init__(self, dataset: AbstractDataset, estimator_class,
                 stationary_margin_model_class, non_stationary_margin_model_class,
                 verbose=False,
                 multiprocessing=False):
        self.verbose = verbose
        self.dataset = dataset
        self.estimator_class = estimator_class
        self.stationary_margin_model_class = stationary_margin_model_class
        self.non_stationary_margin_model_class = non_stationary_margin_model_class
        # Compute a dictionary that maps couple (margin model class, starting point)
        # to the corresponding fitted estimator
        self._starting_point_to_estimator = {}
        # parallelization arguments
        self.multiprocessing = multiprocessing
        self.nb_cores = 7

    def get_estimator(self, starting_point):
        if starting_point not in self._starting_point_to_estimator:
            estimator = self.load_estimator(starting_point)
            self._starting_point_to_estimator[starting_point] = estimator
        return self._starting_point_to_estimator[starting_point]

    def load_estimator(self, starting_point) -> Union[AbstractFullEstimator, AbstractMarginEstimator]:
        margin_model_class = self.stationary_margin_model_class if starting_point is None else self.non_stationary_margin_model_class
        assert starting_point not in self._starting_point_to_estimator
        margin_model = margin_model_class(coordinates=self.dataset.coordinates, starting_point=starting_point)
        estimator = self._load_estimator(margin_model)
        start = time.time()
        estimator.fit()
        duration = time.time() - start
        if self.verbose:
            estimator_name = get_display_name_from_object_type(estimator)
            margin_model_name = get_display_name_from_object_type(margin_model)
            text = 'Fittig {} with margin: {} for starting_point={}\n'.format(estimator_name,
                                                                              margin_model_name,
                                                                              starting_point)
            text += 'Fit took {}s and was {}'.format(round(duration, 1), estimator.result_from_model_fit.convergence)
            print(text)
        return estimator

    def _load_estimator(self, margin_model) -> Union[AbstractFullEstimator, AbstractMarginEstimator]:
        return self.estimator_class(self.dataset, margin_model)

    def get_metric(self, starting_point):
        estimator = self.get_estimator(starting_point)
        metric = estimator.result_from_model_fit.__getattribute__(self.RESULT_ATTRIBUTE_METRIC)
        assert isinstance(metric, float)
        return metric

    def get_mu_coefs(self, starting_point):
        # for the non stationary model gives the mu1 parameters that was fitted
        estimator = self.get_estimator(starting_point)
        margin_function = estimator.function_from_fit  # type: LinearMarginFunction
        assert isinstance(margin_function, LinearMarginFunction)
        mu_coefs = [margin_function.mu_intercept, margin_function.mu1_temporal_trend]
        if self.has_spatial_coordinates:
            mu_coefs += [margin_function.mu_longitude_trend, margin_function.mu_latitude_trend]
        return dict(zip(self.mu_coef_names, mu_coefs))

    @property
    def mu_coef_names(self):
        mu_coef_names = ['mu_intercept', 'mu_temporal']
        if self.has_spatial_coordinates:
            mu_coef_names += ['mu_longitude', 'mu_latitude']
        return mu_coef_names

    @property
    def has_spatial_coordinates(self):
        return self.dataset.coordinates.has_spatial_coordinates

    @property
    def mu_coef_colors(self):
        return ['b', 'c', 'g', 'y', ]

    def visualize(self, ax, complete_analysis=True):
        years = self.years(complete_analysis)

        # Load the estimator only once
        if self.multiprocessing:
            with Pool(self.nb_cores) as p:
                stationary_estimator, *non_stationary_estimators = p.map(self.get_estimator, [None] + years)
        else:
            stationary_estimator = self.get_estimator(None)
            non_stationary_estimators = [self.get_estimator(year) for year in years]
        self._starting_point_to_estimator[None] = stationary_estimator
        for year, non_stationary_estimator in zip(years, non_stationary_estimators):
            self._starting_point_to_estimator[year] = non_stationary_estimator

        # Plot differences
        stationary_metric, *non_stationary_metrics = [self.get_metric(starting_point) for starting_point in
                                                      [None] + years]
        difference = [m - stationary_metric for m in non_stationary_metrics]
        color_difference = 'r'
        label_difference = self.RESULT_ATTRIBUTE_METRIC + ' difference'
        ax.plot(years, difference, color_difference + 'x-', label=label_difference)
        ax.set_ylabel(label_difference, color=color_difference, )

        # Plot zero line
        # years_line = [years[0] -10, years[-1]  + 10]
        # ax.plot(years, [0 for _ in years], 'kx-', label='zero line')
        # Plot significative line corresponding to 0.05 relevance
        alpha = 0.05
        significative_deviance = chi2.ppf(q=1 - alpha, df=1)
        ax.plot(years, [significative_deviance for _ in years], 'mx-', label='significative line')

        all_deviance_data = [significative_deviance] + difference
        min_deviance_data, max_deviance_data = min(all_deviance_data), max(all_deviance_data)

        # Plot the mu1 parameter
        mu_trends = [self.get_mu_coefs(starting_point=year) for year in years]
        mus = {mu_coef_name: [mu_trend[mu_coef_name] for mu_trend in mu_trends] for mu_coef_name in self.mu_coef_names}

        ax2 = ax.twinx()

        for j, (mu_coef_name, mu_coef_values) in enumerate(mus.items()):
            color_mu_coef = self.mu_coef_colors[j]
            if self.verbose:
                print(mu_coef_name, mu_coef_values)
            ax2.plot(years, mu_coef_values, color_mu_coef + 'o-', label=mu_coef_name)
            # ax2.set_ylabel(mu_coef_name, color=color_mu_coef)

        df_mus = pd.DataFrame(mus)
        min_mus, max_mus = df_mus.min().min(), df_mus.max().max()
        min_global, max_global = min(min_deviance_data, min_mus), max(max_deviance_data, max_mus)
        # ax2.set_ylim(min_global, max_global)
        # if min_mus < 0.0 < max_mus:
        #     align_yaxis_on_zero(ax2, ax)

        ax.set_title(self.display_name)
        ax.set_xlabel('starting year for the linear trend of {}'.format(self.mu_coef_names[-1]))
        ax.grid()

        prop = {'size': 5} if not self.has_spatial_coordinates else None
        ax.legend(loc=6, prop=prop)
        ax2.legend(loc=7, prop=prop)

    def years(self, complete_analysis=True):
        # Define the year_min and year_max for the starting point
        if complete_analysis:
            year_min, year_max, step = 1960, 1990, 1
            OptimizationConstants.USE_MAXIT = True
        else:
            year_min, year_max, step = 1960, 1990, 5
        years = list(range(year_min, year_max + 1, step))
        return years

    @property
    def display_name(self):
        raise NotImplementedError


class IndependenceLocationTrendTest(AbstractNonStationaryTrendTest):

    def __init__(self, station_name, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         estimator_class=LinearMarginEstimator,
                         stationary_margin_model_class=StationaryTemporalModel,
                         non_stationary_margin_model_class=NonStationaryLocationTemporalModel)
        self.station_name = station_name

    @property
    def display_name(self):
        return self.station_name


class ConditionalIndedendenceLocationTrendTest(AbstractNonStationaryTrendTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         estimator_class=LinearMarginEstimator,
                         stationary_margin_model_class=LinearStationaryMarginModel,
                         non_stationary_margin_model_class=LinearNonStationaryLocationMarginModel)

    @property
    def display_name(self):
        return 'conditional independence'


class MaxStableLocationTrendTest(AbstractNonStationaryTrendTest):

    def __init__(self, max_stable_model, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         estimator_class=FullEstimatorInASingleStepWithSmoothMargin,
                         stationary_margin_model_class=LinearStationaryMarginModel,
                         non_stationary_margin_model_class=LinearNonStationaryLocationMarginModel)
        self.max_stable_model = max_stable_model

    def _load_estimator(self, margin_model) -> AbstractEstimator:
        return self.estimator_class(self.dataset, margin_model, self.max_stable_model)

    @property
    def display_name(self):
        return get_display_name_from_object_type(type(self.max_stable_model))
