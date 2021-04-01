import numpy as np
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ci_method_to_method_name
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict
from extreme_fit.model.utils import r


class ResultFromEvgam(AbstractResultFromExtremes):

    def __init__(self, result_from_fit: robjects.ListVector, param_name_to_dim=None,
                 dim_to_coordinate=None,
                 type_for_mle="GEV") -> None:
        super().__init__(result_from_fit, param_name_to_dim, dim_to_coordinate)
        self.type_for_mle = type_for_mle

    @property
    def aic(self):
        """Compute the aic from the list of parameters in the results,
         find a way to comptue it directly from the result to compare"""
        location = self.name_to_value['location']
        a = np.array(location)
        print(a)
        print(len(a))
        print('here2')
        # 'location', 'logscale', 'shape'
        raise NotImplementedError

    @property
    def log_scale(self):
        return True

    @property
    def margin_coef_ordered_dict(self):
        # print(self.name_to_value.keys())
        # raise NotImplementedError
        values = np.array(self.name_to_value['coefficients'])
        return get_margin_coef_ordered_dict(self.param_name_to_dim, values, self.type_for_mle,
                                            dim_to_coordinate_name=self.dim_to_coordinate)
