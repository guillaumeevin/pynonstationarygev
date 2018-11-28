import pandas as pd
from enum import Enum

import numpy as np
import rpy2
from rpy2.rinterface import RRuntimeError
import rpy2.robjects as robjects

from extreme_estimator.extreme_models.abstract_model import AbstractModel
from extreme_estimator.extreme_models.utils import r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractMaxStableModel(AbstractModel):

    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit, params_sample)
        self.cov_mod = None

    @property
    def cov_mod_param(self):
        return {'cov.mod': self.cov_mod}

    def fitmaxstab(self, maxima_frech: np.ndarray, df_coordinates: pd.DataFrame, fit_marge=False,
                   fit_marge_form_dict=None, margin_start_dict=None) -> dict:
        assert isinstance(maxima_frech, np.ndarray)
        assert isinstance(df_coordinates, pd.DataFrame)
        if fit_marge:
            assert fit_marge_form_dict is not None
            assert margin_start_dict is not None

        # Prepare the data
        data = np.transpose(maxima_frech)

        # Prepare the coord
        df_coordinates = df_coordinates.copy()
        # In the one dimensional case, fitmaxstab isn't working
        # therefore, we treat our 1D coordinate as 2D coordinate on the line y=x, and enforce iso=TRUE
        fitmaxstab_with_one_dimensional_data = len(df_coordinates.columns) == 1
        if fitmaxstab_with_one_dimensional_data:
            assert AbstractCoordinates.COORDINATE_X in df_coordinates.columns
            df_coordinates[AbstractCoordinates.COORDINATE_Y] = df_coordinates[AbstractCoordinates.COORDINATE_X]
        # Give names to columns to enable a specification of the shape of each marginal parameter
        coord = robjects.vectors.Matrix(df_coordinates.values)
        coord.colnames = robjects.StrVector(list(df_coordinates.columns))

        #  Prepare the fit_params (a dictionary containing all additional parameters)
        fit_params = self.cov_mod_param.copy()
        start_dict = self.params_start_fit
        # Remove some parameters that should only be used either in 1D or 2D case, otherwise fitmaxstab crashes
        start_dict = self.remove_unused_parameters(start_dict, fitmaxstab_with_one_dimensional_data)
        if fit_marge:
            start_dict.update(margin_start_dict)
            fit_params.update({k: robjects.Formula(v) for k, v in fit_marge_form_dict.items()})
        if fitmaxstab_with_one_dimensional_data:
            fit_params['iso'] = True
        fit_params['start'] = r.list(**start_dict)
        fit_params['fit.marge'] = fit_marge

        # Run the fitmaxstab in R
        try:
            res = r.fitmaxstab(data=data, coord=coord, **fit_params)  # type: robjects.ListVector
        except RRuntimeError as error:
            raise Exception('Some R exception have been launched at RunTime: \n {}'.format(error.__repr__()))
        # todo: maybe if the convergence was not successful I could try other starting point several times
        # Retrieve the resulting fitted values
        fitted_values = res.rx2('fitted.values')
        fitted_values = {key: fitted_values.rx2(key)[0] for key in fitted_values.names}
        return fitted_values

    def rmaxstab(self, nb_obs: int, coordinates: np.ndarray) -> np.ndarray:
        """
        Return an numpy of maxima. With rows being the stations and columns being the years of maxima
        """
        maxima_frech = np.array(
            r.rmaxstab(nb_obs, coordinates, *list(self.cov_mod_param.values()), **self.params_sample))
        return np.transpose(maxima_frech)

    def remove_unused_parameters(self, start_dict, coordinate_dim):
        return start_dict


class CovarianceFunction(Enum):
    whitmat = 0
    cauchy = 1
    powexp = 2
    bessel = 3


class AbstractMaxStableModelWithCovarianceFunction(AbstractMaxStableModel):

    def __init__(self, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(params_start_fit, params_sample)
        assert covariance_function is not None
        self.covariance_function = covariance_function
        self.default_params_sample = {
            'range': 3,
            'smooth': 0.5,
            'nugget': 0.5
        }
