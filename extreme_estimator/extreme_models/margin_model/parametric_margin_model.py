from abc import ABC

import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.margin_model.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_estimator.extreme_models.result_from_fit import ResultFromFit
from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.utils import safe_run_r_estimator, r, get_coord, \
    get_margin_formula
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ParametricMarginModel(AbstractMarginModel, ABC):

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None):
        self.margin_function_sample = None  # type: ParametricMarginFunction
        self.margin_function_start_fit = None  # type: ParametricMarginFunction
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample)

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> ResultFromFit:
        # The reshaping on the line below is only valid if we have a single observation per spatio-temporal point
        if maxima_gev.shape[1] == 1:
            maxima_gev = maxima_gev.reshape([len(df_coordinates_temp), len(df_coordinates_spat)])
        data = np.transpose(maxima_gev)

        fit_params = get_margin_formula(self.margin_function_start_fit.form_dict)

        # Covariables
        covariables = get_coord(df_coordinates=df_coordinates_spat)
        fit_params['temp.cov'] = get_coord(df_coordinates=df_coordinates_temp)

        # Start parameters
        coef_dict = self.margin_function_start_fit.coef_dict
        fit_params['start'] = r.list(**coef_dict)

        return safe_run_r_estimator(function=r.fitspatgev, use_start=self.use_start_value, data=data,
                                    covariables=covariables, **fit_params)
