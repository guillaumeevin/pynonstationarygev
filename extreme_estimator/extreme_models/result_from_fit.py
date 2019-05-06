from typing import Dict

from rpy2 import robjects

from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef


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


class ResultFromIsmev(ResultFromFit):
    pass

    @property
    def mle(self):
        return self.res['mle']


    @property
    def nllh(self):
        return self.res['nllh']

class ResultFromSpatialExtreme(ResultFromFit):
    """
    Handler from any result with the result of a fit functions from the package Spatial Extreme
    """
    FITTED_VALUES_NAME = 'fitted.values'
    CONVERGENCE_NAME = 'convergence'


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
