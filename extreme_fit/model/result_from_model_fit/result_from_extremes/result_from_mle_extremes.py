import numpy as np

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict


class ResultFromMleExtremes(AbstractResultFromExtremes):

    @property
    def margin_coef_ordered_dict(self):
        values = self.name_to_value['results']
        d = self.get_python_dictionary(values)
        values = {i: param for i, param in enumerate(np.array(d['par']))}
        return get_margin_coef_ordered_dict(self.gev_param_name_to_dim, values)
