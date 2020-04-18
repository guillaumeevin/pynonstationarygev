from enum import Enum

import numpy as np
import pandas as pd
from rpy2.rinterface import RRuntimeWarning
from rpy2.rinterface._rinterface import RRuntimeError

from extreme_fit.model.abstract_model import AbstractModel
from extreme_fit.model.result_from_model_fit.result_from_spatial_extreme import ResultFromSpatialExtreme
from extreme_fit.model.utils import r, get_coord, \
    get_margin_formula_spatial_extreme, SafeRunException, safe_run_r_estimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractMaxStableModel(AbstractModel):

    def __init__(self, use_start_value=False, params_start_fit=None, params_sample=None):
        super().__init__(use_start_value, params_start_fit, params_sample)
        self.cov_mod = None

    @property
    def cov_mod_param(self):
        return {'cov.mod': self.cov_mod}

    def fitmaxstab(self, df_coordinates_spat: pd.DataFrame, df_coordinates_temp: pd.DataFrame = None,
                   data_frech: np.ndarray = None, data_gev: np.ndarray = None,
                   fit_marge=False, fit_marge_form_dict=None, margin_start_dict=None) -> ResultFromSpatialExtreme:
        assert isinstance(df_coordinates_spat, pd.DataFrame)
        if fit_marge:
            assert fit_marge_form_dict is not None
            assert margin_start_dict is not None

        # Prepare the data
        data = data_gev if fit_marge else data_frech
        assert isinstance(data, np.ndarray)
        assert len(df_coordinates_spat) == data.shape[1]

        # Prepare the coord
        df_coordinates_spat = df_coordinates_spat.copy()
        # In the one dimensional case, fitmaxstab isn't working
        # therefore, we treat our 1D coordinate as 2D coordinate on the line y=x, and enforce iso=TRUE
        fitmaxstab_with_one_dimensional_data = len(df_coordinates_spat.columns) == 1
        if fitmaxstab_with_one_dimensional_data:
            assert AbstractCoordinates.COORDINATE_X in df_coordinates_spat.columns
            df_coordinates_spat[AbstractCoordinates.COORDINATE_Y] = df_coordinates_spat[
                AbstractCoordinates.COORDINATE_X]
        # Give names to columns to enable a specification of the shape of each marginal parameter
        coord = get_coord(df_coordinates_spat)

        #  Prepare the fit_params (a dictionary containing all additional parameters)
        fit_params = self.cov_mod_param.copy()
        start_dict = self.params_start_fit
        # Remove some parameters that should only be used either in 1D or 2D case, otherwise fitmaxstab crashes
        start_dict = self.remove_unused_parameters(start_dict, fitmaxstab_with_one_dimensional_data)
        if fit_marge:
            start_dict.update(margin_start_dict)
            margin_formulas = get_margin_formula_spatial_extreme(fit_marge_form_dict)
            fit_params.update(margin_formulas)
        if fitmaxstab_with_one_dimensional_data:
            fit_params['iso'] = True
        fit_params['fit.marge'] = fit_marge

        # Add some temporal covariates
        # Check the shape of the data
        has_temp_cov = df_coordinates_temp is not None and len(df_coordinates_temp) > 0
        if has_temp_cov:
            assert len(df_coordinates_temp) == len(data)
            fit_params['temp.cov'] = get_coord(df_coordinates_temp)

        # Run the fitmaxstab in R
        res = safe_run_r_estimator(function=r.fitmaxstab, data=data, coord=coord,
                                   start_dict=start_dict,
                                   **fit_params)
        return ResultFromSpatialExtreme(res)

    def rmaxstab(self, nb_obs: int, coordinates_values: np.ndarray,
                 use_rmaxstab_with_2_coordinates=False) -> np.ndarray:
        """
        Return an numpy of maxima. With rows being the stations and columns being the years of maxima
        """
        if use_rmaxstab_with_2_coordinates and coordinates_values.shape[1] > 2:
            # When we have more than 2 coordinates, then sample based on the 2 first coordinates only
            coordinates_values = coordinates_values[:, :2]

        try:
            maxima_frech = np.array(
                r.rmaxstab(nb_obs, coordinates_values, *list(self.cov_mod_param.values()), **self.params_sample))
        except (RRuntimeError, RRuntimeWarning) as e:
            raise SafeRunException('\n\nSome R exception have been launched at RunTime: \n{} \n{}'.format(e.__repr__(),
                                   'To sample from 3D data we advise to set the function argument '
                                   '"use_rmaxstab_with_2_coordinates" to True'))

        return np.transpose(maxima_frech)

    def remove_unused_parameters(self, start_dict, coordinate_dim):
        return start_dict


class CovarianceFunction(Enum):
    whitmat = 0
    cauchy = 1
    powexp = 2
    bessel = 3


class AbstractMaxStableModelWithCovarianceFunction(AbstractMaxStableModel):

    def __init__(self, use_start_value=False, params_start_fit=None, params_sample=None,
                 covariance_function: CovarianceFunction = None):
        super().__init__(use_start_value, params_start_fit, params_sample)
        assert covariance_function is not None
        self.covariance_function = covariance_function
        self.default_params = {
            'range': 3,
            'smooth': 0.5,
            'nugget': 0.5
        }
