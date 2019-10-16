from abc import ABC

from cached_property import cached_property

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.estimator.utils import load_margin_function
from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMarginEstimator(AbstractEstimator, ABC):

    def __init__(self, dataset: AbstractDataset):
        super().__init__(dataset)
        assert self.dataset.maxima_gev() is not None


class LinearMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel):
        super().__init__(dataset)
        assert isinstance(margin_model, LinearMarginModel)
        self.margin_model = margin_model

    def _fit(self) -> AbstractResultFromModelFit:
        maxima_gev_specialized = self.dataset.maxima_gev_for_spatial_extremes_package(self.train_split)
        df_coordinates_spat = self.dataset.coordinates.df_spatial_coordinates(self.train_split)
        df_coordinates_temp = self.dataset.coordinates.df_temporal_coordinates_for_fit(split=self.train_split,
                                                                                       starting_point=self.margin_model.starting_point)
        return self.margin_model.fitmargin_from_maxima_gev(data=maxima_gev_specialized,
                                                                            df_coordinates_spat=df_coordinates_spat,
                                                                            df_coordinates_temp=df_coordinates_temp)

    @cached_property
    def margin_function_from_fit(self) -> LinearMarginFunction:
        return load_margin_function(self, self.margin_model)
