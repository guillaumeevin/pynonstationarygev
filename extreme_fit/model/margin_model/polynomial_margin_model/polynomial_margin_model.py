import itertools

from cached_property import cached_property
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.parametric_margin_function import ParametricMarginFunction
from extreme_fit.function.margin_function.polynomial_margin_function import PolynomialMarginFunction
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef, PolynomialCoef
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import GumbelTemporalModel
from extreme_fit.model.margin_model.polynomial_margin_model.functions_utils import \
    load_param_name_to_polynomial_all_coef
from extreme_fit.model.margin_model.utils import MarginFitMethod
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class PolynomialMarginModel(AbstractTemporalLinearMarginModel):

    def __init__(self, *args, max_degree=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_degree = max_degree

    @cached_property
    def margin_function(self) -> PolynomialMarginFunction:
        return super().margin_function

    def load_margin_function(self, param_name_to_list_dim_and_degree=None):
        # Assert the order of list of dim and degree, to match the order of the form dict,
        # i.e. 1) spatial individual terms 2) combined terms 3) temporal individual terms
        for param_name, list_dim_and_degree in param_name_to_list_dim_and_degree.items():
            dims = [d for d, m in list_dim_and_degree]
            assert all([isinstance(d, int) or isinstance(d, tuple) for d in dims])
            if self.coordinates.has_spatial_coordinates and self.coordinates.idx_x_coordinates in dims:
                assert dims.index(self.coordinates.idx_x_coordinates) == 0
            if self.coordinates.has_temporal_coordinates and self.coordinates.idx_temporal_coordinates in dims:
                assert dims.index(self.coordinates.idx_temporal_coordinates) == len(dims) - 1
        # Assert that the degree are inferior to the max degree
        for list_dim_and_degree in param_name_to_list_dim_and_degree.values():
            for _, max_degree in list_dim_and_degree:
                assert max_degree <= self.max_degree, 'Max degree (={}) specified is too high'.format(max_degree)
        # Load param_name_to_polynomial_all_coef
        param_name_to_polynomial_all_coef = load_param_name_to_polynomial_all_coef(
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
            all_individual_dims = self.coordinates.coordinates_dims
            combinations_of_two_dims = list(itertools.combinations(all_individual_dims, 2))
            dims = all_individual_dims + combinations_of_two_dims
            for dim in dims:
                for degree in range(self.max_degree + 1):
                    param_name_and_dim_and_degree_to_coef[(param_name, dim, degree)] = default_slope
        return param_name_and_dim_and_degree_to_coef

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
