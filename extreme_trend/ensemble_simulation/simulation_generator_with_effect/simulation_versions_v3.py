import numpy as np
from numpy.random import beta

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.independent_margin_function import IndependentMarginFunction
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractSimulationForSnowLoadAt1500(AbstractSimulationWithEffects):

    @property
    def summary_parameter(self):
        return "_{}_{}_{}".format(self.shift_mean,
                                     self.shift_std,
                                     self.alpha_ray)

    @property
    def shift_mean(self):
        return self.average_bias_reference[0] / 100

    @property
    def alpha_ray(self):
        return 5

    @property
    def shift_std(self):
        return self.average_bias_reference[1] / 100

    @property
    def average_bias_reference(self):
        raise NotImplementedError

    @property
    def shift_location(self):
        gev_params = self.gev_params_at_zero
        a = (gev_params.g(1) - 1) * gev_params.scale / gev_params.shape
        res1 = (a + gev_params.location) * self.shift_mean
        res2 = a * self.shift_scale
        res = res1 - res2
        res /= gev_params.location
        return res

    @property
    def shift_scale(self):
        return self.shift_std

    @property
    def location_at_zero(self):
        return 2.77

    @property
    def scale_at_zero(self):
        return 1.30

    @property
    def shape_at_zero(self):
        return -0.105

    @property
    def gev_params_at_zero(self):
        return GevParams(self.location_at_zero, self.scale_at_zero, self.shape_at_zero)

    def load_margin_function(self) -> IndependentMarginFunction:
        gev_params_shifted = GevParams(self.location_at_zero * (1 + self.shift_location),
                                       self.scale_at_zero * (1 + self.shift_scale),
                                       self.shape_at_zero)
        relative_change_in_mean = (
                                          gev_params_shifted.mean - self.gev_params_at_zero.mean) / self.gev_params_at_zero.mean
        assert np.isclose(relative_change_in_mean, self.shift_mean)
        relative_change_in_std = (gev_params_shifted.std - self.gev_params_at_zero.std) / self.gev_params_at_zero.std
        assert np.isclose(relative_change_in_std, self.shift_std)
        # constant parameters
        coef_dict = dict()
        coef_dict['locCoeff1'] = self.location_at_zero
        scale_at_zero = self.scale_at_zero
        coef_dict['scaleCoeff1'] = np.log(scale_at_zero)
        coef_dict['shapeCoeff1'] = self.shape_at_zero
        # Non stationary effects
        coef_dict['tempCoeffLoc1'] = -0.8 * coef_dict['locCoeff1']
        coef_dict['tempCoeffScale1'] = np.log(1 - 0.57)
        coef_dict['tempCoeffShape1'] = -2.11 * coef_dict['shapeCoeff1']
        # Climatic effects
        param_name_to_climate_coordinates_with_effects = {
            GevParams.LOC: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SCALE: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SHAPE: None,
        }
        param_name_to_ordered_climate_effects = {
            GevParams.LOC: [
                self.shift_location * coef_dict['locCoeff1'] for
                _ in
                range(self.nb_ensemble_member)],
            GevParams.SCALE: [np.log(1 + self.shift_scale) for _ in
                              range(self.nb_ensemble_member)],
            GevParams.SHAPE: [],
        }
        # Load margin function
        margin_function = type(self.margin_function).from_coef_dict(self.coordinates,
                                                                    self.margin_function.param_name_to_dims,
                                                                    coef_dict,
                                                                    log_scale=True,
                                                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                                    param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects)
        return margin_function


class SimulationSnowLoadWithShiftLikeSafran(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 4.381, 7.1338

    @property
    def alpha_ray(self):
        return 5

class SimulationSnowLoadWithShift0And0(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, 0


class SimulationLogScaleWithShift10And0(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 10, 0


class SimulationLogScaleWithShift0And10(AbstractSimulationForSnowLoadAt1500):

    @property
    def average_bias_reference(self):
        return 0, 10
