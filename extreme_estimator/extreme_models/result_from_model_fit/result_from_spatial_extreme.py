from typing import Dict

import numpy as np

from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit


class ResultFromSpatialExtreme(AbstractResultFromModelFit):
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

    @property
    def other_coef_dict(self):
        return {k: v for k, v in self.all_parameters.items() if LinearCoef.COEFF_STR not in k}
