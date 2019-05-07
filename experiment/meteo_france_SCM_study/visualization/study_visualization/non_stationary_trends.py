from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from scipy.stats import chi2
from extreme_estimator.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.linear_margin_model import \
    LinearAllParametersTwoFirstCoordinatesMarginModel, LinearAllTwoStatialCoordinatesLocationLinearMarginModel, \
    LinearStationaryMarginModel, LinearNonStationaryLocationMarginModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractNonStationaryTrendTest(object):
    RESULT_ATTRIBUTE_METRIC = 'deviance'

    def __init__(self, dataset: AbstractDataset, estimator_class,
                 stationary_margin_model_class, non_stationary_margin_model_class):
        self.dataset = dataset
        self.estimator_class = estimator_class
        self.stationary_margin_model_class = stationary_margin_model_class
        self.non_stationary_margin_model_class = non_stationary_margin_model_class

    def load_estimator(self, margin_model) -> AbstractEstimator:
        return self.estimator_class(self.dataset, margin_model)

    def get_metric(self, margin_model_class, starting_point):
        margin_model = margin_model_class(coordinates=self.dataset.coordinates, starting_point=starting_point)
        estimator = self.load_estimator(margin_model)   # type: AbstractEstimator
        estimator.fit()
        metric = estimator.result_from_fit.__getattribute__(self.RESULT_ATTRIBUTE_METRIC)
        assert isinstance(metric, float)
        return metric

    def visualize(self, ax, complete_analysis=True):
        # Define the year_min and year_max for the starting point
        if complete_analysis:
            year_min, year_max, step = 1960, 1990, 1
        else:
            year_min, year_max, step = 1960, 1990, 10
        # Fit the stationary model
        stationary_metric = self.get_metric(self.stationary_margin_model_class, starting_point=None)
        # Fit the non stationary model
        years = list(range(year_min, year_max + 1, step))
        non_stationary_metrics = [self.get_metric(self.non_stationary_margin_model_class, year) for year in years]
        difference = [m - stationary_metric for m in non_stationary_metrics]
        # Plot some lines
        ax.axhline(y=0, xmin=year_min, xmax=year_max)
        # Significative line
        significative_deviance = chi2.ppf(q=0.95, df=1)
        ax.axhline(y=significative_deviance, xmin=year_min, xmax=year_max)
        # todo: plot the line corresponding to the significance 0.05
        ax.plot(years, difference, 'ro-')


class IndependenceLocationTrendTest(AbstractNonStationaryTrendTest):

    def __init__(self, dataset, coordinate_idx):
        pass


class ConditionalIndedendenceLocationTrendTest(AbstractNonStationaryTrendTest):

    def __init__(self, dataset):
        super().__init__(dataset=dataset,
                         estimator_class=LinearMarginEstimator,
                         stationary_margin_model_class=LinearStationaryMarginModel,
                         non_stationary_margin_model_class=LinearNonStationaryLocationMarginModel)


class MaxStableLocationTrendTest(AbstractNonStationaryTrendTest):

    def __init__(self, dataset, max_stable_model):
        super().__init__(dataset=dataset,
                         estimator_class=FullEstimatorInASingleStepWithSmoothMargin,
                         stationary_margin_model_class=LinearStationaryMarginModel,
                         non_stationary_margin_model_class=LinearNonStationaryLocationMarginModel)
        self.max_stable_model = max_stable_model

    def load_estimator(self, margin_model) -> AbstractEstimator:
        return self.estimator_class(self.dataset, margin_model, self.max_stable_model)
