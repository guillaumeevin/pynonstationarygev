from abc import ABC
from itertools import chain

import numpy as np
import pandas as pd
from cached_property import cached_property

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.function.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_spatial_extreme import ResultFromSpatialExtreme
from extreme_fit.model.utils import r, get_coord, \
    get_margin_formula_spatial_extreme, safe_run_r_estimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ParametricMarginModel(AbstractMarginModel, ABC):

    def __init__(self, coordinates: AbstractCoordinates,
                 params_user=None, starting_point=None, params_class=GevParams,
                 fit_method=MarginFitMethod.spatial_extremes_mle,
                 temporal_covariate_for_fit=None,
                 param_name_to_climate_coordinates_with_effects=None,
                 linear_effects=(False, False, False),
                 ):
        """
        :param starting_point: starting coordinate for the temporal trend
        """
        super().__init__(coordinates, params_user, params_class)
        self.fit_method = fit_method
        self.starting_point = starting_point
        self.drop_duplicates = True
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.linear_effects = linear_effects

    @property
    def climate_coordinates_with_effects(self):
        """Return all the effect that are needed to account for in the coordinates"""
        if self.param_name_to_climate_coordinates_with_effects is None:
            return None
        else:
            return self.coordinates.load_full_climate_coordinates_with_effects(
                self.param_name_to_climate_coordinates_with_effects)

    @cached_property
    def margin_function(self) -> ParametricMarginFunction:
        margin_function = super().margin_function
        assert isinstance(margin_function, ParametricMarginFunction)
        return margin_function

    @property
    def param_name_to_list_for_result(self):
        return self.margin_function.param_name_to_dims

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> ResultFromSpatialExtreme:
        assert data.shape[1] == len(df_coordinates_spat)
        if self.fit_method == MarginFitMethod.spatial_extremes_mle:
            return self.fit_from_spatial_extremes(data, df_coordinates_spat, df_coordinates_temp)
        else:
            raise NotImplementedError

    def fit_from_spatial_extremes(self, data, df_coordinates_spat, df_coordinates_temp):
        # Margin formula for fitspatgev
        fit_params = get_margin_formula_spatial_extreme(self.margin_function.form_dict)
        # Covariables
        covariables = get_coord(df_coordinates=df_coordinates_spat)
        fit_params['temp.cov'] = get_coord(df_coordinates=df_coordinates_temp)
        # Start parameters
        coef_dict = self.margin_function.coef_dict
        # fit_params['start'] = r.list(**coef_dict)
        res = safe_run_r_estimator(function=r.fitspatgev, data=data,
                                   start_dict=coef_dict,
                                   covariables=covariables, **fit_params)
        return ResultFromSpatialExtreme(res)
