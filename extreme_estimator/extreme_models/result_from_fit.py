from typing import Dict

import numpy as np
from rpy2 import robjects

from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ResultFromFit(object):


    def __init__(self, result_from_fit: robjects.ListVector) -> None:
        if hasattr(result_from_fit, 'names'):
            self.name_to_value = {name: result_from_fit.rx2(name) for name in result_from_fit.names}
        else:
            self.name_to_value = {}

    @property
    def names(self):
        return self.name_to_value.keys()

    @property
    def all_parameters(self):
        raise NotImplementedError

    @property
    def margin_coef_dict(self):
        raise NotImplementedError

    @property
    def nllh(self):
        raise NotImplementedError

    @property
    def deviance(self):
        raise NotImplementedError


class ResultFromIsmev(ResultFromFit):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim) -> None:
        super().__init__(result_from_fit)
        self.gev_param_name_to_dim = gev_param_name_to_dim

    @property
    def margin_coef_dict(self):
        # Build the Coeff dict from gev_param_name_to_dim
        coef_dict = {}
        i = 0
        mle_values = self.name_to_value['mle']
        for gev_param_name in GevParams.PARAM_NAMES:
            # Add intercept
            intercept_coef_name = LinearCoef.coef_template_str(gev_param_name, LinearCoef.INTERCEPT_NAME).format(1)
            coef_dict[intercept_coef_name] = mle_values[i]
            i += 1
            # Add a potential linear temporal trend
            if gev_param_name in self.gev_param_name_to_dim:
                temporal_coef_name = LinearCoef.coef_template_str(gev_param_name,
                                                                  AbstractCoordinates.COORDINATE_T).format(1)
                coef_dict[temporal_coef_name] = mle_values[i]
                i += 1
        return coef_dict

    @property
    def all_parameters(self):
        return self.margin_coef_dict

    @property
    def nllh(self):
        return self.name_to_value['nllh']


class ResultFromSpatialExtreme(ResultFromFit):
    """
    Handler from any result with the result of a fit functions from the package Spatial Extreme
    """
    FITTED_VALUES_NAME = 'fitted.values'
    CONVERGENCE_NAME = 'convergence'

    @property
    def deviance(self):
        return np.array(self.name_to_value['deviance'])[0]

    @property
    def convergence(self) -> str:
        convergence_value = self.name_to_value[self.CONVERGENCE_NAME]
        return convergence_value[0]

    @property
    def is_convergence_successful(self) -> bool:
        return self.convergence == "successful"

    @property
    def all_parameters(self) -> Dict[str, float]:
        fitted_values = self.name_to_value[self.FITTED_VALUES_NAME]
        return {key: fitted_values.rx2(key)[0] for key in fitted_values.names}

    @property
    def margin_coef_dict(self):
        return {k: v for k, v in self.all_parameters.items() if LinearCoef.COEFF_STR in k}
