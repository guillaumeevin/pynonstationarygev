import numpy as np
from numpy.random import beta

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.independent_margin_function import IndependentMarginFunction
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractSimulationLogScale(AbstractSimulationWithEffects):

    @property
    def summary_parameter(self):
        return "{}_{}_{}_{}_{}_{}".format(self.location_at_zero,
                                       self.scale_at_zero,
                                       self.alpha_non_stationary,
                                       self.alpha_rcm_location,
                                       self.alpha_rcm_scale,
                                       self.shift_rcm)

    @property
    def alpha_rcm_location(self):
        # return 0.2 # this is the bias in the mean
        return 0.1 # this is the bias in the mean

    @property
    def alpha_rcm_scale(self):
        # return 0.3 # this is the bias in the mean & in the std (because the scale parameter participate to both)
        return 0.1 # this is the bias in the mean & in the std (because the scale parameter participate to both)

    @property
    def shift_rcm(self):
        raise NotImplementedError

    @property
    def alpha_non_stationary(self):
        # return 0.5 # it roughly diminish from half on the time series plot
        return 0.1

    def sample_uniform_scale(self, alpha):
        return self._sample_uniform(np.log(1 - alpha), np.log(1 + alpha))

    @property
    def location_at_zero(self):
        return 2.53

    @property
    def scale_at_zero(self):
        return 1.11

    def load_margin_function(self) -> IndependentMarginFunction:
        # constant parameters
        coef_dict = dict()
        # coef_dict['locCoeff1'] = self.location_at_zero
        # scale_at_zero = self.scale_at_zero
        # coef_dict['scaleCoeff1'] = np.log(scale_at_zero)
        coef_dict['locCoeff1'] = 10
        coef_dict['scaleCoeff1'] = 1
        shape = beta(6, 9) - 0.5
        coef_dict['shapeCoeff1'] = shape
        # Non stationary effects
        coef_dict['tempCoeffLoc1'] = self.sample_uniform(self.alpha_non_stationary) * coef_dict['locCoeff1']
        coef_dict['tempCoeffScale1'] = self.sample_uniform_scale(self.alpha_non_stationary) * coef_dict['scaleCoeff1']
        coef_dict['tempCoeffShape1'] = self.sample_uniform(self.alpha_non_stationary) * coef_dict['shapeCoeff1']
        # Climatic effects
        param_name_to_climate_coordinates_with_effects = {
            GevParams.LOC: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SCALE: [AbstractCoordinates.COORDINATE_RCM],
            GevParams.SHAPE: None,
        }
        param_name_to_ordered_climate_effects = {
            GevParams.LOC: [(self.shift_rcm + self.sample_uniform(self.alpha_rcm_location)) * coef_dict['locCoeff1'] for _ in
                            range(self.nb_ensemble_member)],
            GevParams.SCALE: [self.sample_uniform_scale(self.alpha_rcm_scale) * coef_dict['scaleCoeff1'] for _ in
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


class SimulationLogScaleWithShift(AbstractSimulationLogScale):

    @property
    def shift_rcm(self):
        return 0.1

class SimulationLogScaleWithoutShift(AbstractSimulationLogScale):

    @property
    def shift_rcm(self):
        return 0