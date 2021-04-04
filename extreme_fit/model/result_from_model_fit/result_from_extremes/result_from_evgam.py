import numpy as np
from rpy2 import robjects
from rpy2.robjects.pandas2ri import ri2py_dataframe

from extreme_fit.distribution.gev.gev_params import GevParams
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
    def nllh(self):
        """Compute the nllh from the list of parameters in the results,
         find a way to comptue it directly from the result to compare"""
        param_names = ['location', 'logscale', 'shape']
        parameters = [np.array(self.get_python_dictionary(self.name_to_value[param_name])['fitted'])
                      for param_name in param_names]
        # Add maxima
        parameters.append(self.maxima)
        for p in parameters:
            assert len(p) == len(self.maxima)
        # Compute nllh
        nllh = 0
        for loc, log_scale, shape, maximum in zip(*parameters):
            gev_params = GevParams(loc, np.exp(log_scale), shape)
            p = gev_params.density(maximum)
            nllh += -np.log(p)
        return nllh

    @property
    def maxima(self):
        return np.array(self.get_python_dictionary(self.name_to_value['location'])['model'][0])

    @property
    def nb_parameters(self):
        return len(np.array(self.name_to_value['coefficients']))

    @property
    def aic(self):
        return 2 * self.nllh + 2 * self.nb_parameters

    @property
    def bic(self):
        return 2 * self.nllh + np.log(len(self.maxima)) * self.nb_parameters

    @property
    def log_scale(self):
        return True

    @property
    def margin_coef_ordered_dict(self):
        coefficients = np.array(self.name_to_value['coefficients'])
        param_name_to_str_formula = {k: str(v) for k, v in self.get_python_dictionary(self.name_to_value['formula']).items()}
        r_param_names_with_spline = [k for k, v in param_name_to_str_formula.items() if "s(" in v]
        if len(r_param_names_with_spline) == 0:
            return get_margin_coef_ordered_dict(self.param_name_to_dim, coefficients, self.type_for_mle,
                                                dim_to_coordinate_name=self.dim_to_coordinate)
        else:
            # todo we might need to delete some coefficient for the spline param so that it does not create some assertion error
            coef_dict = get_margin_coef_ordered_dict(self.param_name_to_dim, coefficients, self.type_for_mle,
                                                dim_to_coordinate_name=self.dim_to_coordinate)
            spline_param_name_to_dim_knots_and_coefficient = {}
            for r_param_name in r_param_names_with_spline:
                print('here')
                param_name = self.r_param_name_to_param_name[r_param_name]
                dim_knots_and_coefficient = {}
                dims = self.param_name_to_dim[param_name]
                if len(dims) > 1:
                    raise NotImplementedError
                else:
                    dim, max_degree = dims[0]
                    d = self.get_python_dictionary(self.name_to_value[r_param_name])
                    coefficients = np.array(d["coefficients"])
                    smooth = list(d['smooth'])[0]
                    knots = np.array(self.get_python_dictionary(smooth)['knots'])
                    assert len(knots) == len(coefficients) + 1 + max_degree
                    dim_knots_and_coefficient[dim] = (knots, coefficients)
                spline_param_name_to_dim_knots_and_coefficient[param_name] = dim_knots_and_coefficient

            return coef_dict, spline_param_name_to_dim_knots_and_coefficient

    @property
    def r_param_name_to_param_name(self):
        return {
            'location': GevParams.LOC,
            'scale': GevParams.SCALE,
            'shape': GevParams.SHAPE,
        }