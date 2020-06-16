from abc import ABC

import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.utils import load_margin_function
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
                                                                        starting_point=self.margin_model.starting_point,
                                                                        drop_duplicates=self.margin_model.drop_duplicates)

    @property
    def maxima_gev_train(self):
        return self.dataset.maxima_gev_for_spatial_extremes_package(self.train_split)

    @cached_property
    def function_from_fit(self) -> LinearMarginFunction:
        return load_margin_function(self, self.margin_model)

    def nllh(self, split=Split.all):
        nllh = 0
        maxima_values = self.dataset.maxima_gev(split=split)
        coordinate_values = self.dataset.coordinates_values(split=split)
        for maximum, coordinate in zip(maxima_values, coordinate_values):
            assert len(
                maximum) == 1, 'So far, only one observation for each coordinate, but code would be easy to change'
            maximum = maximum[0]
            gev_params = self.function_from_fit.get_params(coordinate, is_transformed=True)
            p = gev_params.density(maximum)
            nllh -= np.log(p)
            assert not np.isinf(nllh)
        return nllh

    def aic(self, split=Split.all):
        return 2 * self.margin_model.nb_params + 2 * self.nllh(split=split)

    def bic(self, split=Split.all):
        n = len(self.dataset.maxima_gev(split=split))
        return np.log(n) * self.margin_model.nb_params + 2 * self.nllh(split=split)

