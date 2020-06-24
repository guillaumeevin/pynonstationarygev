from cached_property import cached_property

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.parametric_margin_function import ParametricMarginFunction
from extreme_fit.function.margin_function.polynomial_margin_function import PolynomialMarginFunction
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef, PolynomialCoef
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import GumbelTemporalModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class PolynomialMarginModel(AbstractTemporalLinearMarginModel):

    def __init__(self, coordinates: AbstractCoordinates, params_user=None, starting_point=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle, nb_iterations_for_bayesian_fit=5000,
                 params_initial_fit_bayesian=None, type_for_MLE="GEV", params_class=GevParams, max_degree=2):
        super().__init__(coordinates, params_user, starting_point, fit_method, nb_iterations_for_bayesian_fit,
                         params_initial_fit_bayesian, type_for_MLE, params_class)
        self.max_degree = max_degree

    @property
    def nb_params(self):
        return sum([c.nb_params for c in self.margin_function.param_name_to_coef.values()])

    @cached_property
    def margin_function(self) -> PolynomialMarginFunction:
        return super().margin_function

    def load_margin_function(self, param_name_to_list_dim_and_degree=None):
        param_name_to_polynomial_all_coef = self.param_name_to_polynomial_all_coef(
            param_name_to_list_dim_and_degree=param_name_to_list_dim_and_degree,
            param_name_and_dim_and_degree_to_default_coef=self.default_params)
        return PolynomialMarginFunction(coordinates=self.coordinates,
                                        param_name_to_coef=param_name_to_polynomial_all_coef,
                                        param_name_to_dim_and_max_degree=param_name_to_list_dim_and_degree,
                                        starting_point=self.starting_point,
                                        params_class=self.params_class)

    @property
    def default_params(self) -> dict:
        default_slope = 0.01
        param_name_and_dim_and_degree_to_coef = {}
        for param_name in self.params_class.PARAM_NAMES:
            for dim in self.coordinates.coordinates_dims:
                for degree in range(self.max_degree + 1):
                    param_name_and_dim_and_degree_to_coef[(param_name, dim, degree)] = default_slope
        return param_name_and_dim_and_degree_to_coef

    def param_name_to_polynomial_all_coef(self, param_name_to_list_dim_and_degree,
                                          param_name_and_dim_and_degree_to_default_coef):
        param_name_to_polynomial_all_coef = {}
        param_names = list(set([e[0] for e in param_name_and_dim_and_degree_to_default_coef.keys()]))
        for param_name in param_names:
            dim_to_polynomial_coef = {}
            for dim, max_degree in param_name_to_list_dim_and_degree.get(param_name, []):
                degree_to_coef = {}
                for (param_name_loop, dim_loop, degree), coef in param_name_and_dim_and_degree_to_default_coef.items():
                    if param_name == param_name_loop and dim == dim_loop and degree <= max_degree:
                        degree_to_coef[degree] = coef
                # print(degree_to_coef, param_name)
                # if len(degree_to_coef) == 0:
                #     degree_to_coef = {0: param_name_and_dim_and_degree_to_default_coef[(param_name, dim, 0)]}
                polynomial_coef = PolynomialCoef(param_name, degree_to_coef=degree_to_coef)
                dim_to_polynomial_coef[dim] = polynomial_coef
            if len(dim_to_polynomial_coef) == 0:
                intercept = param_name_and_dim_and_degree_to_default_coef[(param_name, 0, 0)]
                dim_to_polynomial_coef = None
            else:
                intercept = None
            polynomial_all_coef = PolynomialAllCoef(param_name=param_name,
                                                    dim_to_polynomial_coef=dim_to_polynomial_coef,
                                                    intercept=intercept)
            param_name_to_polynomial_all_coef[param_name] = polynomial_all_coef
        return param_name_to_polynomial_all_coef

    @property
    def param_name_to_list_for_result(self):
        return self.margin_function.param_name_to_dim_and_max_degree


class NonStationaryQuadraticLocationModel(PolynomialMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: [(self.coordinates.idx_temporal_coordinates, 2)]})

    @property
    def mul(self):
        return 2


class NonStationaryQuadraticScaleModel(PolynomialMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SCALE: [(self.coordinates.idx_temporal_coordinates, 2)]})

    @property
    def sigl(self):
        return 2


class NonStationaryQuadraticLocationGumbelModel(GumbelTemporalModel, NonStationaryQuadraticLocationModel):
    pass


class NonStationaryQuadraticScaleGumbelModel(GumbelTemporalModel, NonStationaryQuadraticScaleModel):
    pass
