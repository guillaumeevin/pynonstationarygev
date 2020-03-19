from abc import ABC

from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.utils import load_margin_function, compute_nllh
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.slicer.split import Split


class AbstractMarginEstimator(AbstractEstimator, ABC):

    def __init__(self, dataset: AbstractDataset, **kwargs):
        super().__init__(dataset, **kwargs)
        assert self.dataset.maxima_gev() is not None


class LinearMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel, **kwargs):
        super().__init__(dataset, **kwargs)
        assert isinstance(margin_model, LinearMarginModel)
        self.margin_model = margin_model

    def _fit(self) -> AbstractResultFromModelFit:
        df_coordinates_spat = self.dataset.coordinates.df_spatial_coordinates(self.train_split)
        return self.margin_model.fitmargin_from_maxima_gev(data=self.maxima_gev_train,
                                                           df_coordinates_spat=df_coordinates_spat,
                                                           df_coordinates_temp=self.coordinate_temp)

    @property
    def coordinate_temp(self):
        return self.dataset.coordinates.df_temporal_coordinates_for_fit(split=self.train_split,
                                                                        starting_point=self.margin_model.starting_point)

    @property
    def maxima_gev_train(self):
        return self.dataset.maxima_gev_for_spatial_extremes_package(self.train_split)

    @property
    def nllh(self, split=Split.all):
        assert split is Split.all
        return compute_nllh(self, self.maxima_gev_train, self.coordinate_temp, self.margin_model)

    @cached_property
    def function_from_fit(self) -> LinearMarginFunction:
        return load_margin_function(self, self.margin_model)
