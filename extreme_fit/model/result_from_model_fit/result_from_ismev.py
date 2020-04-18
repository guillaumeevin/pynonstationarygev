import numpy as np
import pandas as pd
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.utils import convertFloatVector_to_float
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict


class ResultFromIsmev(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, param_name_to_dim=None) -> None:
        super().__init__(result_from_fit)
        self.param_name_to_dim = param_name_to_dim

    @property
    def margin_coef_ordered_dict(self):
        return get_margin_coef_ordered_dict(self.param_name_to_dim, self.name_to_value['mle'])

    @property
    def all_parameters(self):
        return self.margin_coef_ordered_dict

    @property
    def nllh(self):
        return convertFloatVector_to_float(self.name_to_value['nllh'])

    @property
    def convergence(self) -> str:
        return convertFloatVector_to_float(self.name_to_value['conv']) == 0

    @property
    def covariance(self):
        return pd.DataFrame(np.array(self.name_to_value['cov']),
                            index=self.margin_coef_ordered_names,
                            columns=self.margin_coef_ordered_names)
