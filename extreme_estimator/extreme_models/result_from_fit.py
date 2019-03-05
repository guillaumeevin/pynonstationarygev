from typing import Dict

from rpy2 import robjects


class ResultFromFit(object):
    """
    Handler from any result with the result of a fit functions from the package Spatial Extreme
    """
    FITTED_VALUES_NAME = 'fitted.values'
    CONVERGENCE_NAME = 'convergence'

    def __init__(self, result_from_fit: robjects.ListVector) -> None:
        if hasattr(result_from_fit, 'names'):
            self.name_to_value = {name: result_from_fit.rx2(name) for name in result_from_fit.names}
        else:
            self.name_to_value = {}

    @property
    def names(self):
        return self.name_to_value.keys()

    @property
    def convergence(self):
        convergence_value = self.name_to_value[self.CONVERGENCE_NAME]
        return convergence_value

    @property
    def fitted_values(self) -> Dict[str, float]:
        fitted_values = self.name_to_value[self.FITTED_VALUES_NAME]
        return {key: fitted_values.rx2(key)[0] for key in fitted_values.names}
