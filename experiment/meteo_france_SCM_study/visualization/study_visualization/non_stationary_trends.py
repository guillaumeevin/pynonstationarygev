import time
from multiprocessing import Pool
from typing import Union

from experiment.meteo_france_SCM_study.visualization.utils import align_yaxis_on_zero
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from scipy.stats import chi2
from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin, AbstractFullEstimator
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator, \
    AbstractMarginEstimator
from extreme_estimator.extreme_models.margin_model.linear_margin_model import \
    LinearAllParametersTwoFirstCoordinatesMarginModel, LinearAllTwoStatialCoordinatesLocationLinearMarginModel, \
    LinearStationaryMarginModel, LinearNonStationaryLocationMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.utils import OptimizationConstants
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from utils import get_display_name_from_object_type


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
        self._margin_model_class_and_starting_point_to_estimator = {}
        # parallelization arguments
        self.multiprocessing = multiprocessing
        self.nb_cores = 7

    def get_estimator(self, margin_model_class, starting_point) -> Union[
        AbstractFullEstimator, AbstractMarginEstimator]:
        if (margin_model_class, starting_point) not in self._margin_model_class_and_starting_point_to_estimator:
            margin_model = margin_model_class(coordinates=self.dataset.coordinates, starting_point=starting_point)
            estimator = self._load_estimator(margin_model)
            start = time.time()
            estimator.fit()
            duration = time.time() - start
            if self.verbose:
                if self.verbose:
                    estimator_name = get_display_name_from_object_type(estimator)
                    margin_model_name = get_display_name_from_object_type(margin_model)
                    text = 'Fittig {} with margin: {} for starting_point={}'.format(estimator_name,
                                                                                             margin_model_name,
                                                                                             starting_point)
                    text += 'Fit took {}s and was {}'.format(round(duration, 1), estimator.result_from_fit.convergence)
                    print(text)
            self._margin_model_class_and_starting_point_to_estimator[(margin_model_class, starting_point)] = estimator
        return self._margin_model_class_and_starting_point_to_estimator[(margin_model_class, starting_point)]

    def _load_estimator(self, margin_model) -> Union[AbstractFullEstimator, AbstractMarginEstimator]:
        return self.estimator_class(self.dataset, margin_model)

    def get_metric(self, starting_point):
        margin_model_class = self.stationary_margin_model_class if starting_point is None else self.non_stationary_margin_model_class
        estimator = self.get_estimator(margin_model_class, starting_point)
        metric = estimator.result_from_fit.__getattribute__(self.RESULT_ATTRIBUTE_METRIC)
        assert isinstance(metric, float)
        return metric

    def get_mu1(self, starting_point):
        # for the non stationary model gives the mu1 parameters that was fitted
        estimator = self.get_estimator(self.non_stationary_margin_model_class, starting_point)
        margin_function = estimator.margin_function_fitted  # type: LinearMarginFunction
        assert isinstance(margin_function, LinearMarginFunction)
        return margin_function.mu1_temporal_trend

    def visualize(self, ax, complete_analysis=True):
        years = self.years(complete_analysis)

        # Compute metric with parallelization or not
        if self.multiprocessing:
            with Pool(self.nb_cores) as p:
                stationary_metric, *non_stationary_metrics = p.map(self.get_metric, [None] + years)
        else:
            stationary_metric = self.get_metric(starting_point=None)
            non_stationary_metrics = [self.get_metric(year) for year in years]

        # Plot differences
        difference = [m - stationary_metric for m in non_stationary_metrics]
        color_difference = 'b'
        ax.plot(years, difference, color_difference + 'o-')
        ax.set_ylabel(self.RESULT_ATTRIBUTE_METRIC + ' difference', color=color_difference)

        # Plot zero line
        # years_line = [years[0] -10, years[-1]  + 10]
        ax.plot(years, [0 for _ in years], 'k-', label='zero line')
        # Plot significative line corresponding to 0.05 relevance
        alpha = 0.05
        significative_deviance = chi2.ppf(q=1 - alpha, df=1)
        ax.plot(years, [significative_deviance for _ in years], 'g-', label='significative line')

        # Plot the mu1 parameter
        mu1_trends = [self.get_mu1(starting_point=year) for year in years]
        ax2 = ax.twinx()
        color_mu1 = 'c'

        if self.verbose:
            print('mu1 trends:', mu1_trends, '\n')
        ax2.plot(years, mu1_trends, color_mu1 + 'o-')
        ax2.set_ylabel('mu1 parameter', color=color_mu1)

        ax.set_xlabel('starting year for the linear trend of mu1')
        if min(mu1_trends) < 0.0 < max(mu1_trends):
            align_yaxis_on_zero(ax, ax2)
        title = self.display_name
        ax.set_title(title)
        ax.legend()

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

    def __init__(self, dataset, coordinate_idx):
        pass


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
