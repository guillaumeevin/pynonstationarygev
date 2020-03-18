from abc import ABC

import numpy as np
import pandas as pd

from extreme_fit.function.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_fit.model.result_from_model_fit.result_from_spatial_extreme import ResultFromSpatialExtreme
from extreme_fit.model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_fit.model.utils import safe_run_r_estimator, r, get_coord, \
    get_margin_formula_spatial_extreme
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ParametricMarginModel(AbstractMarginModel, ABC):

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None, starting_point=None):
        """
        :param starting_point: starting coordinate for the temporal trend
        """
        self.starting_point = starting_point
        self.margin_function_sample = None  # type: ParametricMarginFunction
        self.margin_function_start_fit = None  # type: ParametricMarginFunction
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample)

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> ResultFromSpatialExtreme:
        assert data.shape[1] == len(df_coordinates_spat)

        # Margin formula for fitspatgev
        fit_params = get_margin_formula_spatial_extreme(self.margin_function_start_fit.form_dict)

        # Covariables
        covariables = get_coord(df_coordinates=df_coordinates_spat)
        fit_params['temp.cov'] = get_coord(df_coordinates=df_coordinates_temp)

        # Start parameters
        coef_dict = self.margin_function_start_fit.coef_dict
        fit_params['start'] = r.list(**coef_dict)

        res = safe_run_r_estimator(function=r.fitspatgev, use_start=self.use_start_value, data=data,
                                   covariables=covariables, **fit_params)
        return ResultFromSpatialExtreme(res)
