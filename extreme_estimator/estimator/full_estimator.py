from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractFullEstimator(AbstractEstimator):
    pass


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
        maxima_frech = AbstractMarginModel.gev2frech(maxima_gev=self.dataset.maxima_gev,
                                                     coordinates=self.dataset.coordinates,
                                                     margin_function=self.margin_estimator.margin_function_fitted)
        # Update maxima frech field through the dataset object
        self.dataset.maxima_frech = maxima_frech
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
        self.smooth_margin_function_to_fit = margin_model.margin_function_start_fit

    def _fit(self):
        # todo: specify the shape of the margin
        # Estimate the margin
        self.max_stable_params_fitted = self.max_stable_model.fitmaxstab(
            maxima_frech=self.dataset.maxima_frech,
            df_coordinates=self.dataset.df_coordinates,
            fit_marge=True,
            fit_marge_form_dict=self.smooth_margin_function_to_fit.fit_marge_form_dict,
            margin_start_dict=self.smooth_margin_function_to_fit.margin_start_dict
        )


class PointwiseAndThenUnitaryMsp(AbstractFullEstimator):
    pass


class StochasticExpectationMaximization(AbstractFullEstimator):
    pass


class INLAgoesExtremes(AbstractFullEstimator):
    pass
