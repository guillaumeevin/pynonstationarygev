from numpy.random import beta

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.independent_margin_function import IndependentMarginFunction
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class SimulationVersion1(AbstractSimulationWithEffects):

    def load_margin_function(self) -> IndependentMarginFunction:
        # Sample the non-stationary parameters
        coef_dict = dict()
        coef_dict['locCoeff1'] = 10
        coef_dict['scaleCoeff1'] = 1
        shape = beta(6, 9) - 0.5
        coef_dict['shapeCoeff1'] = shape
        coef_dict['tempCoeffLoc1'] = self.sample_around(coef_dict['locCoeff1'])
        coef_dict['tempCoeffScale1'] = self.sample_around(coef_dict['scaleCoeff1'])
        coef_dict['tempCoeffShape1'] = self.sample_around(coef_dict['shapeCoeff1'])
        # Climatic effects
        param_name_to_climate_coordinates_with_effects = {
            GevParams.LOC: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SCALE: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SHAPE: None,
        }
        param_name_to_ordered_climate_effects = {
            GevParams.LOC: [self.sample_around(coef_dict['locCoeff1']) for _ in range(self.nb_ensemble_member)],
            GevParams.SCALE: [self.sample_around(coef_dict['scaleCoeff1']) for _ in range(self.nb_ensemble_member)],
            GevParams.SHAPE: [],
        }
        # Load margin function
        margin_function = type(self.margin_function).from_coef_dict(self.coordinates,
                                                                    self.margin_function.param_name_to_dims,
                                                                    coef_dict,
                                                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                                    param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects)
        return margin_function


class SimulationVersion2(AbstractSimulationWithEffects):
    """just add some factor 1.2 in the effect"""

    def load_margin_function(self) -> IndependentMarginFunction:
        # Sample the non-stationary parameters
        coef_dict = dict()
        coef_dict['locCoeff1'] = 10
        coef_dict['scaleCoeff1'] = 1
        shape = beta(6, 9) - 0.5
        coef_dict['shapeCoeff1'] = shape
        coef_dict['tempCoeffLoc1'] = self.sample_uniform(0.1) * coef_dict['locCoeff1']
        coef_dict['tempCoeffScale1'] = self.sample_uniform(0.1) * coef_dict['scaleCoeff1']
        coef_dict['tempCoeffShape1'] = self.sample_uniform(0.1) * coef_dict['shapeCoeff1']
        # Climatic effects
        param_name_to_climate_coordinates_with_effects = {
            GevParams.LOC: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SCALE: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SHAPE: None,
        }
        param_name_to_ordered_climate_effects = {
            GevParams.LOC: [(0.1 + self.sample_uniform(0.15)) * coef_dict['locCoeff1'] for _ in range(self.nb_ensemble_member)],
            GevParams.SCALE: [self.sample_uniform(0.1) * coef_dict['scaleCoeff1'] for _ in range(self.nb_ensemble_member)],
            GevParams.SHAPE: [],
        }
        # Load margin function
        margin_function = type(self.margin_function).from_coef_dict(self.coordinates,
                                                                    self.margin_function.param_name_to_dims,
                                                                    coef_dict,
                                                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                                    param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects)
        return margin_function
