import numpy as np
import pandas as pd
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.utils import r


class AbstractResultFromExtremes(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None) -> None:
        super().__init__(result_from_fit)
        self.gev_param_name_to_dim = gev_param_name_to_dim

    def load_dataframe_from_r_matrix(self, name):
        r_matrix = self.name_to_value[name]
        return pd.DataFrame(np.array(r_matrix), columns=r.colnames(r_matrix))
