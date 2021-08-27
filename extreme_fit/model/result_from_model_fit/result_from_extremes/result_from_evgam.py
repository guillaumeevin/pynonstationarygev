import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.pandas2ri import ri2py_dataframe
from scipy.interpolate import make_interp_spline

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ci_method_to_method_name
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict
from extreme_fit.model.utils import r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ResultFromEvgam(AbstractResultFromExtremes):

    def __init__(self, result_from_fit: robjects.ListVector, param_name_to_dim=None,
                 dim_to_coordinate=None,
                 type_for_mle="GEV",
                 param_name_to_name_of_the_climatic_effects=None,
                 param_name_to_climate_coordinates_with_effects=None,
                 linear_effects=(False, False, False),
                 ) -> None:
        super().__init__(result_from_fit, param_name_to_dim, dim_to_coordinate)
        self.linear_effects = linear_effects
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.param_name_to_name_of_the_climatic_effects = param_name_to_name_of_the_climatic_effects
        self.type_for_mle = type_for_mle

    @property
    def param_name_to_name_of_the_climatic_effects_to_load_margin_function(self):
        return self.param_name_to_name_of_the_climatic_effects

    @property
    def param_name_to_climate_coordinates_with_effects_to_load_margin_function(self):
        return self.param_name_to_climate_coordinates_with_effects

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
        for j, (loc, log_scale, shape, maximum) in enumerate(zip(*parameters)):
            scale = np.exp(log_scale)
            gev_params = GevParams(loc, scale, shape)
            p = gev_params.density(maximum)
            nllh += -np.log(p)
        return nllh

    def get_gev_params_from_result(self, idx):
        param_names = ['location', 'logscale', 'shape']
        loc, log_scale, shape = [np.array(self.get_python_dictionary(self.name_to_value[param_name])['fitted'])[idx]
                                 for param_name in param_names]
        return GevParams(loc, np.exp(log_scale), shape)

    @property
    def maxima(self):
        return np.array(self.get_python_dictionary(self.name_to_value['location'])['model'][0])

    @property
    def nb_parameters(self):
        # return len(np.array(self.name_to_value['coefficients']))  + self.nb_knots
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
        param_name_to_str_formula = {k: str(v) for k, v in
                                     self.get_python_dictionary(self.name_to_value['formula']).items()}
        r_param_names_with_spline = [k for k, v in param_name_to_str_formula.items() if "s(" in v]
        r_param_names_with_spline = [k if k != "scale" else "logscale" for k in r_param_names_with_spline]
        if len(r_param_names_with_spline) == 0:
            return get_margin_coef_ordered_dict(self.param_name_to_dims, coefficients, self.type_for_mle,
                                                dim_to_coordinate_name=self.dim_to_coordinate,
                                                param_name_to_name_of_the_climatic_effects=self.param_name_to_name_of_the_climatic_effects,
                                                linear_effects=self.linear_effects)
        else:
            # Compute spline param_name to dim_to_knots_and_coefficients
            spline_param_name_to_dim_knots_and_coefficient = {}
            param_names_with_spline = [self.r_param_name_to_param_name[r_param_name] for r_param_name in
                                       r_param_names_with_spline]
            for param_name, r_param_name in zip(param_names_with_spline, r_param_names_with_spline):
                dim_knots_and_coefficient = self.compute_dim_to_knots_and_coefficient(param_name, r_param_name)
                spline_param_name_to_dim_knots_and_coefficient[param_name] = dim_knots_and_coefficient
            # Modify the param_name_to_dim for the spline variables (to not miss any variable)
            param_name_to_dims = self.param_name_to_dims.copy()
            for param_name in param_names_with_spline:
                new_dims = []
                for dim, _ in param_name_to_dims[param_name]:
                    nb_coefficients_for_param_name = len(
                        spline_param_name_to_dim_knots_and_coefficient[param_name][dim][1])
                    new_dim = (dim, nb_coefficients_for_param_name - 1)
                    new_dims.append(new_dim)
                param_name_to_dims[param_name] = new_dims
            # Extract the coef list
            coefficients = [self.load_coefficients(r_param_name,
                                                   self.load_knots(r_param_name)) for r_param_name in self.r_param_names]
            coefficients = np.concatenate(coefficients)
            assert len(coefficients) == len(np.array(self.name_to_value['coefficients']))
            coef_dict = get_margin_coef_ordered_dict(param_name_to_dims, coefficients, self.type_for_mle,
                                                     dim_to_coordinate_name=self.dim_to_coordinate,
                                                     param_name_to_name_of_the_climatic_effects=self.param_name_to_name_of_the_climatic_effects)
            return coef_dict, spline_param_name_to_dim_knots_and_coefficient

    def compute_dim_to_knots_and_coefficient(self, param_name, r_param_name):
        dim_knots_and_coefficient = {}
        dims = self.param_name_to_dims[param_name]
        if len(dims) > 1:
            raise NotImplementedError
        else:
            dim, max_degree = dims[0]
            knots = self.load_knots(r_param_name)
            # Load the coordinates
            data = np.array(self.name_to_value["data"])
            x = data[1]
            y = np.array(self.get_python_dictionary(self.name_to_value[r_param_name])['fitted'])
            if (len(data) > 2) and (self.param_name_to_climate_coordinates_with_effects[param_name] is not None):
                x_climatic = data[2:]
                y = self.remove_effects_from_y_from_all_climate_model(x_climatic, y, r_param_name, param_name, knots)
            x_for_interpolation = knots[1:-1]
            x_short, y_short = self.extract_x_and_y(x, x_for_interpolation, y)
            spline = make_interp_spline(x_short, y_short, k=1, t=knots)
            coefficients = spline.c
            assert len(knots) == len(coefficients) + 1 + max_degree
            dim_knots_and_coefficient[dim] = (knots, coefficients)
        return dim_knots_and_coefficient

    def extract_x_and_y(self, x, x_for_interpolation, y):
        # For the time covariate, the distance will be zero for the closer year
        # For the temperature covariate, the distance will be minimal for the closer covariate
        index = []
        for x_to_find in x_for_interpolation:
            distances = np.abs(x - x_to_find)
            closer_index = np.argmin(distances)
            index.append(closer_index)
        x_short = np.array([x[i] for i in index])
        y_short = np.array([y[i] for i in index])
        return x_short, y_short

    def remove_effects_from_y_from_all_climate_model(self, x_climatic, y, r_param_name, param_name, knots):
        # Run the remove effect
        y = y.copy()
        name_of_the_climatic_effects = self.param_name_to_name_of_the_climatic_effects[param_name]
        assert name_of_the_climatic_effects is not None
        # Extract potential a subpart of x_climatic
        climate_coordinates_with_effects = self.param_name_to_climate_coordinates_with_effects[param_name]
        assert climate_coordinates_with_effects is not None
        if len(climate_coordinates_with_effects) == 1:
            if climate_coordinates_with_effects[0] == AbstractCoordinates.COORDINATE_GCM:
                x_climatic = x_climatic[:len(name_of_the_climatic_effects)]
            else:
                x_climatic = x_climatic[-len(name_of_the_climatic_effects):]
        # Load the coefficient correspond to the effect from the last climate model
        coefficients = self.load_coefficients(r_param_name, knots)
        effects_coefficients = coefficients[-len(name_of_the_climatic_effects):]
        assert len(effects_coefficients) == len(name_of_the_climatic_effects) == len(x_climatic)
        df_coordinates = pd.DataFrame(x_climatic.transpose(), columns=name_of_the_climatic_effects)
        for j, effect_coef in enumerate(effects_coefficients):
            ind = df_coordinates.iloc[:, j] == 1.0
            assert len(ind) == len(y)
            y[ind.values] -= effect_coef
        return y

    def load_knots(self, r_param_name):
        try:
            d = self.get_python_dictionary(self.name_to_value[r_param_name])
            smooth = list(d['smooth'])[0]
            knots = np.array(self.get_python_dictionary(smooth)['knots'])
        except (IndexError, KeyError):
            knots = []
        return knots

    def load_coefficients(self, r_param_name, knots):
        try:
            d = self.get_python_dictionary(self.name_to_value[r_param_name])
            coefficients = np.array(d['coefficients'])
            if len(self.load_knots(r_param_name)) > 0:
                # the coefficients are not in the expected order, so we reorder them.
                coefficients = list(coefficients)
                nb_spline_coef_minus_1 = len(knots) - 3
                new_coefficients = coefficients[:1] + coefficients[-nb_spline_coef_minus_1:] + coefficients[1:-nb_spline_coef_minus_1]
                assert len(new_coefficients) == len(coefficients)
                coefficients = np.array(new_coefficients)
        except (IndexError, KeyError):
            coefficients = []
        return coefficients

    @property
    def nb_knots(self):
        return sum([len(self.load_knots(r_param_name)) for r_param_name in self.r_param_name_to_param_name.keys()])

    @property
    def r_param_names(self):
        return ['location', 'logscale', 'shape']

    @property
    def r_param_name_to_param_name(self):
        return {
            'location': GevParams.LOC,
            'logscale': GevParams.SCALE,
            'shape': GevParams.SHAPE,
        }
