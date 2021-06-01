from abc import ABC
import numpy.testing as npt

import numpy as np
import pandas as pd
from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh, \
    compute_nllh_with_multiprocessing_for_large_samples
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMarginEstimator(AbstractEstimator, ABC):

    def __init__(self, dataset: AbstractDataset, **kwargs):
        super().__init__(dataset, **kwargs)
        assert self.dataset.maxima_gev is not None


class LinearMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel, **kwargs):
        super().__init__(dataset, **kwargs)
        assert isinstance(margin_model, LinearMarginModel)
        self.margin_model = margin_model

    def _fit(self) -> AbstractResultFromModelFit:
        return self.margin_model.fitmargin_from_maxima_gev(data=self.data,
                                                           df_coordinates_spat=self.df_coordinates_spat,
                                                           df_coordinates_temp=self.df_coordinates_temp)

    @property
    def data(self):
        return self._maxima_gev

    @property
    def _maxima_gev(self):
        if self.margin_model.fit_method == MarginFitMethod.spatial_extremes_mle:
            return self.dataset.maxima_gev_for_spatial_extremes_package
        else:
            return self.dataset.maxima_gev

    @property
    def nb_data(self):
        return len(self.data)

    @property
    def df_coordinates_spat(self):
        return self.dataset.coordinates.df_spatial_coordinates(drop_duplicates=self.margin_model.drop_duplicates)

    @property
    def df_coordinates_temp(self):
        coordinates = self.dataset.coordinates
        df_coordinates_temp = self.load_coordinates_temp(coordinates)
        return df_coordinates_temp

    @property
    def df_coordinates_for_fit(self):
        return pd.concat([self.df_coordinates_spat, self.df_coordinates_temp], axis=1)

    def load_coordinates_temp(self, coordinates):
        assert coordinates.gcm_rcm_couple_as_pseudo_truth == self.dataset.coordinates.gcm_rcm_couple_as_pseudo_truth, \
            "you should set the gcm rcm couple as pseudo truth similarly"
        df_coordinates_temp = coordinates.df_temporal_coordinates_for_fit(
            temporal_covariate_for_fit=self.margin_model.temporal_covariate_for_fit,
            starting_point=self.margin_model.starting_point,
            drop_duplicates=self.margin_model.drop_duplicates,
            climate_coordinates_with_effects=self.margin_model.climate_coordinates_with_effects)
        return df_coordinates_temp

    @cached_property
    def margin_function_from_fit(self) -> LinearMarginFunction:
        return load_margin_function(self, self.margin_model)

    @property
    def coordinates_for_nllh(self):
        return self.df_coordinates_for_fit.values

    @cached_property
    def nllh(self):
        maxima_values = self.dataset.maxima_gev
        coordinate_values = self.coordinates_for_nllh
        nllh = compute_nllh_with_multiprocessing_for_large_samples(coordinate_values, maxima_values,
                                                                   self.margin_function_from_fit)
        npt.assert_almost_equal(self.result_from_model_fit.nllh, nllh, decimal=0)
        return nllh

    @property
    def deviance(self):
        return 2 * self.nllh

    @property
    def aic(self):
        aic = 2 * self.nb_params + 2 * self.nllh
        npt.assert_almost_equal(self.result_from_model_fit.aic, aic, decimal=0)
        return aic

    @property
    def nb_params(self):
        nb_params = self.margin_function_from_fit.nb_params_for_margin_function
        nb_params += self.margin_function_from_fit.nb_params_for_climate_effects
        if isinstance(self.margin_model, AbstractTemporalLinearMarginModel) and self.margin_model.is_gumbel_model:
            nb_params -= 1
        return nb_params

    @property
    def bic(self):
        return np.log(self.nb_data) * self.nb_params + 2 * self.nllh

    @property
    def aicc(self):
        additional_term = 2 * self.nb_params * (self.nb_params + 1) / (self.nb_data - self.nb_params - 1)
        return self.aic + additional_term

    def sorted_empirical_standard_gumbel_quantiles(self, coordinate_for_filter=None):
        sorted_empirical_quantiles = []
        maxima_values = self.dataset.maxima_gev
        coordinate_values = self.df_coordinates_for_fit.values
        for maximum, coordinate in zip(maxima_values, coordinate_values):
            if coordinate_for_filter is not None:
                assert len(coordinate) == len(coordinate_for_filter)
                keep = any([(f is not None) and (c == f) for c, f in zip(coordinate, coordinate_for_filter)])
                if not keep:
                    continue
            gev_param = self.margin_function_from_fit.get_params(
                coordinate=coordinate,
                is_transformed=False)
            # Take the first and unique maximum
            maximum = maximum[0]
            if isinstance(maximum, np.ndarray):
                maximum = maximum[0]
            maximum_standardized = gev_param.gumbel_standardization(maximum)
            sorted_empirical_quantiles.append(maximum_standardized)
        sorted_empirical_quantiles = sorted(sorted_empirical_quantiles)
        return sorted_empirical_quantiles

    def coordinate_values_to_maxima_from_standard_gumbel_quantiles(self, standard_gumbel_quantiles):
        coordinate_values_to_maxima = {}
        coordinate_values = self.df_coordinates_for_fit.values
        assert len(standard_gumbel_quantiles) == len(coordinate_values)
        for quantile, coordinate in zip(standard_gumbel_quantiles, coordinate_values):
            gev_param = self.margin_function_from_fit.get_params(
                coordinate=coordinate,
                is_transformed=False)
            maximum = gev_param.gumbel_inverse_standardization(quantile)
            coordinate_values_to_maxima[tuple(coordinate)] = np.array([maximum])
        return coordinate_values_to_maxima
