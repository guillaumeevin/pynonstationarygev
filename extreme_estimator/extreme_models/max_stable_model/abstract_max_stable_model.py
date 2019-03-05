from enum import Enum

import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.abstract_model import AbstractModel
from extreme_estimator.extreme_models.result_from_fit import ResultFromFit
from extreme_estimator.extreme_models.utils import r, safe_run_r_estimator, get_coord, \
    get_margin_formula
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractMaxStableModel(AbstractModel):

    def __init__(self, use_start_value=False, params_start_fit=None, params_sample=None):
        super().__init__(use_start_value, params_start_fit, params_sample)
        self.cov_mod = None

    @property
    def cov_mod_param(self):
        return {'cov.mod': self.cov_mod}

    def fitmaxstab(self, df_coordinates: pd.DataFrame, maxima_frech: np.ndarray = None, maxima_gev: np.ndarray = None,
                   fit_marge=False, fit_marge_form_dict=None, margin_start_dict=None) -> ResultFromFit:
        assert isinstance(df_coordinates, pd.DataFrame)
        if fit_marge:
            assert fit_marge_form_dict is not None
            assert margin_start_dict is not None

        # Prepare the data
        maxima = maxima_gev if fit_marge else maxima_frech
        assert isinstance(maxima, np.ndarray)
        assert len(df_coordinates) == len(maxima), 'Coordinates and observations sizes should match,' \
                                                   'check that the same split was used for both objects, \n' \
                                                   'df_coordinates size: {}, data size {}'.format(len(df_coordinates),
                                                                                                  len(maxima))
        data = np.transpose(maxima)

        # Prepare the coord
        df_coordinates = df_coordinates.copy()
        # In the one dimensional case, fitmaxstab isn't working
        # therefore, we treat our 1D coordinate as 2D coordinate on the line y=x, and enforce iso=TRUE
        fitmaxstab_with_one_dimensional_data = len(df_coordinates.columns) == 1
        if fitmaxstab_with_one_dimensional_data:
            assert AbstractCoordinates.COORDINATE_X in df_coordinates.columns
            df_coordinates[AbstractCoordinates.COORDINATE_Y] = df_coordinates[AbstractCoordinates.COORDINATE_X]
        # Give names to columns to enable a specification of the shape of each marginal parameter
        coord = get_coord(df_coordinates)

        #  Prepare the fit_params (a dictionary containing all additional parameters)
        fit_params = self.cov_mod_param.copy()
        start_dict = self.params_start_fit
        # Remove some parameters that should only be used either in 1D or 2D case, otherwise fitmaxstab crashes
        start_dict = self.remove_unused_parameters(start_dict, fitmaxstab_with_one_dimensional_data)
        if fit_marge:
            start_dict.update(margin_start_dict)
            margin_formulas = get_margin_formula(fit_marge_form_dict)
            fit_params.update(margin_formulas)
        if fitmaxstab_with_one_dimensional_data:
            fit_params['iso'] = True
        fit_params['start'] = r.list(**start_dict)
        fit_params['fit.marge'] = fit_marge

        # Run the fitmaxstab in R
        return safe_run_r_estimator(function=r.fitmaxstab, use_start=self.use_start_value, data=data, coord=coord, **fit_params)

    def rmaxstab(self, nb_obs: int, coordinates_values: np.ndarray) -> np.ndarray:
        """
        Return an numpy of maxima. With rows being the stations and columns being the years of maxima
        """

        maxima_frech = np.array(
            r.rmaxstab(nb_obs, coordinates_values, *list(self.cov_mod_param.values()), **self.params_sample))
        return np.transpose(maxima_frech)

    def remove_unused_parameters(self, start_dict, coordinate_dim):
        return start_dict


class CovarianceFunction(Enum):
    whitmat = 0
    cauchy = 1
    powexp = 2
    bessel = 3


class AbstractMaxStableModelWithCovarianceFunction(AbstractMaxStableModel):

    def __init__(self, use_start_value=False, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(use_start_value, params_start_fit, params_sample)
        assert covariance_function is not None
        self.covariance_function = covariance_function
        self.default_params_sample = {
            'range': 3,
            'smooth': 0.5,
            'nugget': 0.5
        }
