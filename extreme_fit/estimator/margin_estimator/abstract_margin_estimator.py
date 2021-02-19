from abc import ABC
import numpy.testing as npt

import numpy as np
import pandas as pd
from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.margin_model.utils import MarginFitMethod
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
        data = self.data(self.train_split)
        df_coordinate_temp = self.df_coordinates_temp(self.train_split)
        df_coordinate_spat = self.df_coordinates_spat(self.train_split)
        return self.margin_model.fitmargin_from_maxima_gev(data=data,
                                                           df_coordinates_spat=df_coordinate_spat,
                                                           df_coordinates_temp=df_coordinate_temp)

    def data(self, split):
        return self._maxima_gev(split)

    def _maxima_gev(self, split):
        if self.margin_model.fit_method == MarginFitMethod.spatial_extremes_mle:
            return self.dataset.maxima_gev_for_spatial_extremes_package(split)
        else:
            return self.dataset.maxima_gev(split)

    def df_coordinates_spat(self, split):
        return self.dataset.coordinates.df_spatial_coordinates(split=split,
                                                               drop_duplicates=self.margin_model.drop_duplicates)

    def df_coordinates_temp(self, split):
        return self.dataset.coordinates.df_temporal_coordinates_for_fit(split=split,
                                                                        temporal_covariate_for_fit=self.margin_model.temporal_covariate_for_fit,
                                                                        starting_point=self.margin_model.starting_point,
                                                                        drop_duplicates=self.margin_model.drop_duplicates)

    @cached_property
    def function_from_fit(self) -> LinearMarginFunction:
        return load_margin_function(self, self.margin_model)

    def nllh(self, split=Split.all):
        nllh = 0
        maxima_values = self.dataset.maxima_gev(split=split)
        df = pd.concat([self.df_coordinates_spat(split=split), self.df_coordinates_temp(split=split)], axis=1)
        coordinate_values = df.values
        for maximum, coordinate in zip(maxima_values, coordinate_values):
            assert len(maximum) == 1, \
                'So far, only one observation for each coordinate, but code would be easy to change'
            maximum = maximum[0]
            gev_params = self.function_from_fit.get_params(coordinate, is_transformed=True)
            p = gev_params.density(maximum)
            nllh -= np.log(p)
            assert not np.isinf(nllh)
        return nllh

    def sorted_empirical_standard_gumbel_quantiles(self, split=Split.all, coordinate_for_filter=None):
        sorted_empirical_quantiles = []
        maxima_values = self.dataset.maxima_gev(split=split)
        coordinate_values = self.dataset.df_coordinates(split=split).values
        for maximum, coordinate in zip(maxima_values, coordinate_values):
            if coordinate_for_filter is not None:
                assert len(coordinate) == len(coordinate_for_filter)
                keep = any([(f is not None) and (c == f) for c, f in zip(coordinate, coordinate_for_filter)])
                if not keep:
                    continue
            gev_param = self.function_from_fit.get_params(
                coordinate=coordinate,
                is_transformed=False)
            maximum_standardized = gev_param.gumbel_standardization(maximum[0])
            sorted_empirical_quantiles.append(maximum_standardized)
        sorted_empirical_quantiles = sorted(sorted_empirical_quantiles)
        return sorted_empirical_quantiles

    def coordinate_values_to_maxima_from_standard_gumbel_quantiles(self, standard_gumbel_quantiles, split=Split.all):
        coordinate_values_to_maxima = {}
        coordinate_values = self.dataset.df_coordinates(split=split).values
        assert len(standard_gumbel_quantiles) == len(coordinate_values)
        for quantile, coordinate in zip(standard_gumbel_quantiles, coordinate_values):
            gev_param = self.function_from_fit.get_params(
                coordinate=coordinate,
                is_transformed=False)
            maximum = gev_param.gumbel_inverse_standardization(quantile)
            coordinate_values_to_maxima[tuple(coordinate)] = np.array([maximum])
        return coordinate_values_to_maxima

    def deviance(self, split=Split.all):
        return 2 * self.nllh(split=split)

    def aic(self, split=Split.all):
        aic = 2 * self.margin_model.nb_params + 2 * self.nllh(split=split)
        npt.assert_almost_equal(self.result_from_model_fit.aic, aic, decimal=0)
        return aic

    def n(self, split=Split.all):
        return len(self.dataset.maxima_gev(split=split))

    def bic(self, split=Split.all):
        return np.log(self.n(split=split)) * self.margin_model.nb_params + 2 * self.nllh(split=split)
