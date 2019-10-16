from cached_property import cached_property

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.estimator.max_stable_estimator.abstract_max_stable_estimator import MaxStableEstimator
from extreme_estimator.estimator.utils import load_margin_function
from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractFullEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset):
        super().__init__(dataset)


class SmoothMarginalsThenUnitaryMsp(AbstractFullEstimator):

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel,
                 max_stable_model: AbstractMaxStableModel):
        super().__init__(dataset)
        # Instantiate the two associated estimators
        self.margin_estimator = LinearMarginEstimator(dataset=dataset, margin_model=margin_model)
        self.max_stable_estimator = MaxStableEstimator(dataset=dataset, max_stable_model=max_stable_model)

    def fit(self):
        # Estimate the margin parameters
        self.margin_estimator.fit()
        # Compute the maxima_frech
        maxima_gev_train = self.dataset.maxima_gev(split=self.train_split)
        maxima_frech = AbstractMarginModel.gev2frech(maxima_gev=maxima_gev_train,
                                                     coordinates_values=self.dataset.coordinates_values(
                                                         self.train_split),
                                                     margin_function=self.margin_estimator.margin_function_from_fit)
        # Update maxima frech field through the dataset object
        self.dataset.set_maxima_frech(maxima_frech, split=self.train_split)
        # Estimate the max stable parameters
        self.max_stable_estimator.fit()

    """
    To clean things, I could let an abstract Estimator whose only function is fit
    then create a sub class abstract Estimator One Model, where the function _fit exists and always return something
    then here full estimator class will be a sub class of abstract estimator, with two  abstract Estimator One Model as attributes
    """


class FullEstimatorInASingleStep(AbstractFullEstimator):
    pass


class FullEstimatorInASingleStepWithSmoothMargin(AbstractFullEstimator):
    """The method of Gaume, check if its method is in a single step or not"""

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel,
                 max_stable_model: AbstractMaxStableModel):
        super().__init__(dataset)
        self.max_stable_model = max_stable_model
        self.linear_margin_model = margin_model
        assert isinstance(self.margin_function_start_fit, LinearMarginFunction)

    @property
    def margin_function_start_fit(self):
        return self.linear_margin_model.margin_function_start_fit

    @property
    def df_coordinates_spat(self):
        return self.dataset.coordinates.df_spatial_coordinates(self.train_split)

    @property
    def df_coordinates_temp(self):
        return self.dataset.coordinates.df_temporal_coordinates_for_fit(split=self.train_split,
                                                                        starting_point=self.linear_margin_model.starting_point)

    def _fit(self):
        # Estimate both the margin and the max-stable structure
        return self.max_stable_model.fitmaxstab(
            data_gev=self.dataset.maxima_gev_for_spatial_extremes_package(self.train_split),
            df_coordinates_spat=self.df_coordinates_spat,
            df_coordinates_temp=self.df_coordinates_temp,
            fit_marge=True,
            fit_marge_form_dict=self.linear_margin_model.margin_function_start_fit.form_dict,
            margin_start_dict=self.linear_margin_model.margin_function_start_fit.coef_dict
        )

    @cached_property
    def margin_function_from_fit(self) -> LinearMarginFunction:
        return load_margin_function(self, self.linear_margin_model)


class PointwiseAndThenUnitaryMsp(AbstractFullEstimator):
    pass


class StochasticExpectationMaximization(AbstractFullEstimator):
    pass


class INLAgoesExtremes(AbstractFullEstimator):
    pass
