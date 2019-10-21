from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.utils import convertFloatVector_to_float, get_margin_coef_dict
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.utils import convertFloatVector_to_float


class ResultFromIsmev(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None) -> None:
        super().__init__(result_from_fit)
        self.gev_param_name_to_dim = gev_param_name_to_dim

    @property
    def margin_coef_dict(self):
        return get_margin_coef_dict(self.gev_param_name_to_dim, self.name_to_value['mle'])

    @property
    def all_parameters(self):
        return self.margin_coef_dict

    @property
    def nllh(self):
        return convertFloatVector_to_float(self.name_to_value['nllh'])

    @property
    def deviance(self):
        return - 2 * self.nllh

    @property
    def convergence(self) -> str:
        return convertFloatVector_to_float(self.name_to_value['conv']) == 0
