from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import SmoothMarginEstimator
from extreme_estimator.estimator.max_stable_estimator.abstract_max_stable_estimator import MaxStableEstimator
from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearMarginModel
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
        self.margin_estimator = SmoothMarginEstimator(dataset=dataset, margin_model=margin_model)
        self.max_stable_estimator = MaxStableEstimator(dataset=dataset, max_stable_model=max_stable_model)

    def _fit(self):
        # Estimate the margin parameters
        self.margin_estimator.fit()
        # Compute the maxima_frech
        maxima_gev_train = self.dataset.maxima_gev(split=self.train_split)
        maxima_frech = AbstractMarginModel.gev2frech(maxima_gev=maxima_gev_train,
                                                     coordinates_values=self.dataset.coordinates_values(
                                                         self.train_split),
                                                     margin_function=self.margin_estimator.margin_function_fitted)
        # Update maxima frech field through the dataset object
        self.dataset.set_maxima_frech(maxima_frech, split=self.train_split)
        # Estimate the max stable parameters
        self.max_stable_estimator.fit()


class FullEstimatorInASingleStep(AbstractFullEstimator):
    pass


class FullEstimatorInASingleStepWithSmoothMargin(AbstractFullEstimator):
    """The method of Gaume, check if its method is in a single step or not"""

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel,
                 max_stable_model: AbstractMaxStableModel):
        super().__init__(dataset)
        self.max_stable_model = max_stable_model
        self.linear_margin_model = margin_model
        self.linear_margin_function_to_fit = self.linear_margin_model.margin_function_start_fit
        assert isinstance(self.linear_margin_function_to_fit, LinearMarginFunction)

    def _fit(self):
        # Estimate both the margin and the max-stable structure
        self._result_from_fit = self.max_stable_model.fitmaxstab(
            maxima_gev=self.dataset.maxima_gev(split=self.train_split),
            df_coordinates=self.dataset.df_coordinates(split=self.train_split),
            fit_marge=True,
            fit_marge_form_dict=self.linear_margin_function_to_fit.form_dict,
            margin_start_dict=self.linear_margin_function_to_fit.coef_dict
        )
        # Create the fitted margin function
        self.extract_fitted_models_from_fitted_params(self.linear_margin_function_to_fit, self.fitted_values)


class PointwiseAndThenUnitaryMsp(AbstractFullEstimator):
    pass


class StochasticExpectationMaximization(AbstractFullEstimator):
    pass


class INLAgoesExtremes(AbstractFullEstimator):
    pass
