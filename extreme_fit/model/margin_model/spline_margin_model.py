from typing import Dict, List

from extreme_fit.model.margin_model.margin_function.spline_margin_function import SplineMarginFunction
from extreme_fit.model.margin_model.param_function.abstract_coef import AbstractCoef
from extreme_fit.model.margin_model.param_function.spline_coef import SplineCoef, KnotCoef, \
    PolynomialCoef
from extreme_fit.model.margin_model.parametric_margin_model import ParametricMarginModel
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class SplineMarginModel(ParametricMarginModel):

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None):
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample)

    def load_margin_functions(self, gev_param_name_to_dims: Dict[str, List[int]] = None,
                              gev_param_name_to_coef: Dict[str, AbstractCoef] = None,
                              gev_param_name_to_nb_knots: Dict[str, int] = None,
                              degree=3):
        # Default parameters
        # todo: for the default parameters: take inspiration from the linear_margin_model
        # also implement the class method thing
        if gev_param_name_to_dims is None:
            gev_param_name_to_dims = {gev_param_name: self.coordinates.coordinates_dims
                                      for gev_param_name in GevParams.PARAM_NAMES}
        if gev_param_name_to_coef is None:
            gev_param_name_to_coef = {}
            for gev_param_name in GevParams.PARAM_NAMES:
                knot_coef = KnotCoef(gev_param_name)
                polynomial_coef = PolynomialCoef(gev_param_name)
                dim_to_polynomial_coef = {dim: polynomial_coef for dim in self.coordinates.coordinates_dims}
                spline_coef = SplineCoef(gev_param_name, knot_coef, dim_to_polynomial_coef)
                gev_param_name_to_coef[gev_param_name] = spline_coef
        if gev_param_name_to_nb_knots is None:
            gev_param_name_to_nb_knots = {gev_param_name: 2 for gev_param_name in GevParams.PARAM_NAMES}

        # Load sample coef
        self.margin_function_sample = SplineMarginFunction(coordinates=self.coordinates,
                                                           gev_param_name_to_dims=gev_param_name_to_dims,
                                                           gev_param_name_to_coef=gev_param_name_to_coef,
                                                           gev_param_name_to_nb_knots=gev_param_name_to_nb_knots,
                                                           degree=degree)
        # Load start fit coef
        self.margin_function_start_fit = SplineMarginFunction(coordinates=self.coordinates,
                                                              gev_param_name_to_dims=gev_param_name_to_dims,
                                                              gev_param_name_to_coef=gev_param_name_to_coef,
                                                              gev_param_name_to_nb_knots=gev_param_name_to_nb_knots,
                                                              degree=degree)


class ConstantSplineMarginModel(SplineMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims: Dict[str, List[int]] = None,
                              gev_param_name_to_coef: Dict[str, AbstractCoef] = None,
                              gev_param_name_to_nb_knots: Dict[str, int] = None, degree=3):
        super().load_margin_functions({}, gev_param_name_to_coef, gev_param_name_to_nb_knots,
                                      degree)


class Degree1SplineMarginModel(SplineMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims: Dict[str, List[int]] = None,
                              gev_param_name_to_coef: Dict[str, AbstractCoef] = None,
                              gev_param_name_to_nb_knots: Dict[str, int] = None, degree=3):
        super().load_margin_functions(gev_param_name_to_dims, gev_param_name_to_coef, gev_param_name_to_nb_knots,
                                      degree=1)
