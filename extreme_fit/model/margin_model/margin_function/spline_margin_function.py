from typing import Dict, List

import numpy as np

from extreme_fit.model.margin_model.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_fit.model.margin_model.param_function.abstract_coef import AbstractCoef
from extreme_fit.model.margin_model.param_function.param_function import AbstractParamFunction, \
    SplineParamFunction
from extreme_fit.model.margin_model.param_function.spline_coef import SplineCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class SplineMarginFunction(ParametricMarginFunction):
    """
    -gev_param_name_to_dims maps each GEV parameters to its correspond knot dimensions.
        For instance, dims = [1,2] means the knot will be realized with 2D knots
        dims = [1] means the knot will lie only on the first axis

    """

    COEF_CLASS = SplineCoef

    def __init__(self, coordinates: AbstractCoordinates, gev_param_name_to_dims: Dict[str, List[int]],
                 gev_param_name_to_coef: Dict[str, AbstractCoef],
                 gev_param_name_to_nb_knots: Dict[str, int],
                 degree=3):
        self.gev_param_name_to_coef = None  # type: Dict[str, SplineCoef]
        # Attributes specific for SplineMarginFunction
        self.gev_param_name_to_nb_knots = gev_param_name_to_nb_knots
        assert degree % 2 == 1
        self.degree = degree
        super().__init__(coordinates, gev_param_name_to_dims, gev_param_name_to_coef)


    def compute_knots(self, dims, nb_knots) -> np.ndarray:
        """Define the knots as the quantiles"""
        return np.quantile(a=self.coordinates.df_all_coordinates.iloc[:, dims], q=np.linspace(0, 1, nb_knots+2)[1:-1])

    @property
    def form_dict(self) -> Dict[str, str]:
        """
        3 examples of potential form dict:
            loc.form <- y ~ rb(locations[,1], knots = knots, degree = 3, penalty = .5)
            scale.form <- y ~ rb(locations[,2], knots = knots2, degree = 3, penalty = .5)
            shape.form <- y ~ rb(locations, knots = knots_tot, degree = 3, penalty = .5)
        """
        pass

    def load_specific_param_function(self, gev_param_name) -> AbstractParamFunction:
        dims = self.gev_param_name_to_dims[gev_param_name]
        coef = self.gev_param_name_to_coef[gev_param_name]
        nb_knots = self.gev_param_name_to_nb_knots[gev_param_name]
        knots = self.compute_knots(dims, nb_knots)
        return SplineParamFunction(dims=dims, degree=self.degree, spline_coef=coef, knots=knots)










